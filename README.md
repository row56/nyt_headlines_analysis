## News Reader and Analysis of Political Coverage, Sentiment and Bias in New York Times Headlines

by Robert Wienröder

Screencast here: https://cloud.bht-berlin.de/index.php/s/CSPpqwDTJLYrAcf

See workflow.html or workflow.ipynb for the workflow of the whole project



#### Methodology

This project tried to analyze the political coverage in New York Times headlines over a 70-day period, with a focus on the U.S. presidential campaigns. The methodology consisted of the following steps:

1. Data Collection: Headlines from the New York Times website were scraped for a sample of the past 70 days.

2. Categorization: A machine learning model from Hugging Face, pre-trained on news headline classification tasks, was used to categorize the collected headlines.

3. Candidate-Specific Analysis: The dataset was filtered twice to isolate political headlines containing the keywords "Trump" or "Harris", the presidency candidates of the two major U.S. political parties.

4. Coverage and Sentiment Analysis: a. Coverage analysis was performed to compare the frequency of headlines mentioning each candidate. b. Sentiment analysis was performed on the candidate-specific headlines.

5. Bias Analysis: A separate machine learning model, also from Hugging Face, was employed to conduct a bias analysis on the entire subset of headlines classified as political, to detect any overall ideological tendencies in the New York Times' politics section.
