import csv
import random
from collections import Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define your categories
categories = [
    'Politics', 'U.S.', 'World', 'Sports', 'Science & Tech', 'Business & Economy',
    'Arts & Entertainment', 'Lifestyle', 'Artículos en Español', 'not an article'
]

# 0. Politics
# 1. U.S.
# 2. World
# 3. Sports
# 4. Science & Tech
# 5. Business & Economy
# 6. Arts & Entertainment
# 7. Lifestyle
# 8. Artículos en Español
# 9. not an article

# Pre-defined category mapping
# Replace the empty brackets with the index of your desired category
# Leave the brackets empty to skip the category
category_mapping = {
    0: categories[6], # ARTS
    1: categories[6], # ARTS & CULTURE
    2: categories[0], # BLACK VOICES
    3: categories[5], # BUSINESS
    4: categories[1], # COLLEGE
    5: categories[6], # COMEDY
    6: categories[1], # CRIME
    7: categories[6], # CULTURE & ARTS
    8: categories[7], # DIVORCE
    9: categories[0], # EDUCATION
    10: categories[6], # ENTERTAINMENT
    11: categories[0], # ENVIRONMENT
    12: categories[9], # FIFTY
    13: categories[7], # FOOD & DRINK
    14: categories[7], # GOOD NEWS
    15: categories[9], # GREEN
    16: categories[7], # HEALTHY LIVING
    17: categories[7], # HOME & LIVING
    18: categories[9], # IMPACT
    19: categories[0], # LATINO VOICES
    20: categories[0], # MEDIA
    21: categories[5], # MONEY
    22: categories[7], # PARENTING
    23: categories[7], # PARENTS
    24: categories[0], # POLITICS
    25: categories[0], # QUEER VOICES
    26: categories[0], # RELIGION
    27: categories[4], # SCIENCE
    28: categories[3], # SPORTS
    29: categories[7], # STYLE
    30: categories[7], # STYLE & BEAUTY
    31: categories[7], # TASTE
    32: categories[4], # TECH
    33: categories[2], # THE WORLDPOST
    34: categories[7], # TRAVEL
    35: categories[1], # U.S. NEWS
    36: categories[7], # WEDDINGS
    37: categories[7], # WEIRD NEWS
    38: categories[7], # WELLNESS
    39: categories[0], # WOMEN
    40: categories[2], # WORLD NEWS
    41: categories[2], # WORLDPOST
}

def categorize_articles(articles, model, tokenizer, category_mapping):
    results = []
    for article in articles:
        inputs = tokenizer(article['title'], truncation=True, padding=True, max_length=model.config.max_position_embeddings, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        
        if predicted_class in category_mapping and category_mapping[predicted_class] is not None:
            article['predicted_category'] = category_mapping[predicted_class]
        else:
            article['predicted_category'] = 'Uncategorized'
        
        results.append(article)
    return results

def get_user_feedback(sample):
    for article in sample:
        print(f"\nTitle: {article['title']}")
        print(f"Predicted category: {article['predicted_category']}")
        print("Categories:")
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category}")
        
        while True:
            user_input = input("Press Enter if correct, or enter the number of the correct category: ").strip()
            if user_input == "":
                article['verified_category'] = article['predicted_category']
                break
            try:
                category_index = int(user_input) - 1
                if 0 <= category_index < len(categories):
                    article['verified_category'] = categories[category_index]
                    break
                else:
                    print("Invalid category number. Please try again.")
            except ValueError:
                print("Please enter a valid number or press Enter.")
    return sample

def train_model(training_data, model, tokenizer):
    dataset = Dataset.from_pandas(pd.DataFrame(training_data))
    
    def tokenize_and_encode(examples):
        tokenized = tokenizer(examples["title"], padding="max_length", truncation=True, max_length=model.config.max_position_embeddings, return_tensors="pt")
        tokenized['labels'] = [categories.index(cat) for cat in examples['verified_category']]
        return tokenized

    tokenized_datasets = dataset.map(tokenize_and_encode, batched=True, remove_columns=dataset.column_names)
    tokenized_datasets.set_format("torch")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    train_dataloader = DataLoader(tokenized_datasets, batch_size=8, shuffle=True)

    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            
            loss = loss_fn(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    return model

def evaluate_performance(categorized_articles, verified_samples):
    correct = 0
    total = len(verified_samples)
    category_performance = {cat: {'correct': 0, 'total': 0} for cat in categories}

    for article in verified_samples:
        predicted = article['predicted_category']
        verified = article['verified_category']
        if predicted == verified:
            correct += 1
            category_performance[verified]['correct'] += 1
        category_performance[verified]['total'] += 1

    overall_accuracy = correct / total if total > 0 else 0
    print(f"\nOverall Accuracy: {overall_accuracy:.2%}")
    
    print("\nCategory-wise Performance:")
    for cat, perf in category_performance.items():
        if perf['total'] > 0:
            accuracy = perf['correct'] / perf['total']
            print(f"{cat}: {accuracy:.2%} ({perf['correct']}/{perf['total']})")

    all_categories = [article['predicted_category'] for article in categorized_articles]
    category_distribution = Counter(all_categories)
    total_articles = len(categorized_articles)

    print("\nCategory Distribution in Full Dataset:")
    for cat, count in category_distribution.items():
        percentage = count / total_articles * 100
        print(f"{cat}: {percentage:.2f}% ({count}/{total_articles})")

def main():
    model_name = "dima806/news-category-classifier-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    with open('nyt_articles.csv', 'r', newline='', encoding='utf-8') as infile:
        articles = list(csv.DictReader(infile))

    categorized_articles = categorize_articles(articles, model, tokenizer, category_mapping)
    verified_articles = {}

    verified_samples = []
    continue_tuning = True
    round = 1

    while continue_tuning:
        print(f"\nFine-tuning round {round}")
        
        while True:
            try:
                num_samples = int(input(f"Enter the number of samples for round {round}: "))
                if num_samples > 0 and num_samples <= len(categorized_articles):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(categorized_articles)}")
            except ValueError:
                print("Please enter a valid number")

        if round == 1:
            sample = random.sample(categorized_articles, num_samples)
        else:
            weak_categories = input("Enter weak categories (comma-separated), or press Enter for random sampling: ").split(',')
            if weak_categories == ['']:
                sample = random.sample(categorized_articles, num_samples)
            else:
                weak_samples = [article for article in categorized_articles 
                                if article['predicted_category'] in weak_categories]
                sample = random.sample(weak_samples, min(num_samples, len(weak_samples)))
                
                if len(sample) < num_samples:
                    remaining_samples = random.sample([a for a in categorized_articles if a not in sample], 
                                                      num_samples - len(sample))
                    sample.extend(remaining_samples)

        verified_sample = get_user_feedback(sample)
        verified_samples.extend(verified_sample)

        for article in verified_sample:
            verified_articles[article['title']] = article

        model = train_model(verified_samples, model, tokenizer)

        categorized_articles = [
            verified_articles[article['title']] if article['title'] in verified_articles
            else categorize_articles([article], model, tokenizer, category_mapping)[0]
            for article in articles
        ]

        evaluate_performance(categorized_articles, verified_samples)

        continue_tuning = input("Do you want to do another round of fine-tuning? (y/n): ").lower() == 'y'
        round += 1

    final_categorized_articles = categorize_articles(articles, model, tokenizer, category_mapping)

    with open('final_categorized_articles.csv', 'w', newline='', encoding='utf-8') as outfile:
        all_fields = set()
        for article in final_categorized_articles:
            all_fields.update(article.keys())
        
        fieldnames = list(all_fields)
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_categorized_articles)

    print("Categorization complete. Results saved to 'final_categorized_articles.csv'")

    # Save the model
    model_save_path = "fine_tuned_news_categorization_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")    

if __name__ == "__main__":
    main()









