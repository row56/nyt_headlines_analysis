import csv
import logging
from flask import Flask, render_template, redirect, url_for, abort

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define categories with correct capitalization
categories = [
    'Politics', 'U.S.', 'World', 'Sports', 'Science & Tech', 'Business & Economy',
    'Arts & Entertainment', 'Lifestyle'
]

def capitalize_category(category):
    if category.lower() == 'u.s.':
        return 'U.S.'
    words = category.split()
    return ' '.join(word.capitalize() for word in words)

def load_articles():
    articles = []
    try:
        with open('categorized_nyt_articles.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            logger.debug(f"CSV columns: {fieldnames}")
            
            for i, row in enumerate(reader):
                if 'predicted_category' not in row:
                    logger.warning(f"Row {i} is missing 'predicted_category' key. Row data: {row}")
                    row['predicted_category'] = 'Uncategorized'
                if row['predicted_category'].lower() != 'not an article':
                    row['predicted_category'] = capitalize_category(row['predicted_category'])
                    if row['predicted_category'] in categories:
                        articles.append(row)
        
        logger.debug(f"Loaded {len(articles)} articles")
        logger.debug(f"Sample article: {articles[0] if articles else 'No articles'}")
        logger.debug(f"Categories found: {set(article.get('predicted_category', 'N/A') for article in articles)}")
    except Exception as e:
        logger.error(f"Error loading articles: {str(e)}")
    return articles

@app.route('/')
def index():
    articles = load_articles()
    return render_template('index.html', articles=articles, categories=categories)

@app.route('/category/<category>')
def category(category):
    try:
        articles = load_articles()
        category_capitalized = capitalize_category(category)
        filtered_articles = [article for article in articles if article.get('predicted_category') == category_capitalized]
        if not filtered_articles:
            logger.warning(f"No articles found for category: {category_capitalized}")
        return render_template('category.html', articles=filtered_articles, category=category_capitalized, categories=categories)
    except Exception as e:
        logger.error(f"Error in category route: {str(e)}")
        return render_template('error.html', error=str(e)), 500

@app.route('/article/<path:url>')
def article(url):
    return redirect(url)

@app.route('/debug/articles')
def debug_articles():
    articles = load_articles()
    return render_template('debug_articles.html', articles=articles)

if __name__ == '__main__':
    app.run(debug=True)
    logger.info('Flask app is starting in debug mode')