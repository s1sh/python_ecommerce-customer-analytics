#!/usr/bin/env python
# coding: utf-8

# # Text Mining & Sentiment Analytics

# <h3>Imports & Setup</h3>




import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Natural Language Processing (NLP) & Sentiment Analysis
from textblob import TextBlob  

from wordcloud import WordCloud, STOPWORDS  # Generates word cloud visual representations and provides default stopword lists
import nltk  # Natural Language Toolkit
from nltk.corpus import stopwords  # Imports predefined lists of common non-informative words (e.g., "the", "is", "at") to filter out
import re  # Regular Expressions module used for text cleaning (e.g., removing punctuation, URLs, numbers, and special characters)

# Download NLTK stopwords if needed
nltk.download('stopwords', quiet=True)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Setup complete for Text Mining & Sentiment Analytics!")


# <h3>Load Processed Dataset</h3>




data_path = '../data/processed/cleaned_reviews.csv'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Run 01_data_prep.ipynb first!")

df = pd.read_csv(data_path)

# Filter out missing reviews for clean NLP scoring
df_nlp = df[df['review_text'] != 'Missing'].copy()
print(f"Loaded {len(df_nlp):,} valid consumer text reviews for sentiment analysis.")


# <h3>Sentiment Polarity & Subjectivity Computation using TextBlob</h3>



# Compute TextBlob Polarity (-1.0 to +1.0) and Subjectivity (0.0 to 1.0)
df_nlp['polarity'] = df_nlp['review_text'].apply(lambda text: TextBlob(str(text)).sentiment.polarity)
df_nlp['subjectivity'] = df_nlp['review_text'].apply(lambda text: TextBlob(str(text)).sentiment.subjectivity)

# Assign categorical sentiment labels
def get_sentiment_category(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df_nlp['sentiment_category'] = df_nlp['polarity'].apply(get_sentiment_category)

print("Sentiment Score Summary:")
print(df_nlp[['polarity', 'subjectivity', 'sentiment_category']].describe())


# <h3>Sentiment Category Distribution vs. Star Ratings</h3>




fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Plot 1: Sentiment Breakdown Counts
sns.countplot(
    data=df_nlp, 
    x='sentiment_category', 
    order=['Positive', 'Neutral', 'Negative'],
    hue='sentiment_category',
    palette=['#2ecc71', '#95a5a6', '#e74c3c'], 
    ax=axes[0]
)
axes[0].set_title('Derived Sentiment Breakdown (TextBlob)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Review Count')

# Plot 2: Sentiment Polarity vs Numeric Rating Correlation
sns.boxplot(data=df_nlp, x='rating', y='polarity',hue='rating', palette='Blues', ax=axes[1])
axes[1].set_title('Sentiment Polarity across Star Ratings (1–5)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Star Rating')
axes[1].set_ylabel('Text Polarity (-1 to +1)')

plt.tight_layout()
plt.show()


# <h3>Comparative Word Clouds (Positive vs. Negative Reviews)</h3>




# Setup custom stop words list
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(['amazon', 'device', 'product', 'bought', 'use','got', 'one', 'tablet','great','good','love','like','bad','work'])

pos_text = " ".join(review for review in df_nlp[df_nlp['sentiment_category'] == 'Positive']['review_text'])
neg_text = " ".join(review for review in df_nlp[df_nlp['sentiment_category'] == 'Negative']['review_text'])

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Positive Word Cloud
wordcloud_pos = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords, max_words=100).generate(pos_text)
axes[0].imshow(wordcloud_pos, interpolation='bilinear')
axes[0].set_title('Positive Reviews Word Cloud', fontsize=16, fontweight='bold', color='green')
axes[0].axis('off')

# Negative Word Cloud
wordcloud_neg = WordCloud(width=800, height=400, background_color='black', stopwords=custom_stopwords, max_words=100, colormap='autumn').generate(neg_text)
axes[1].imshow(wordcloud_neg, interpolation='bilinear')
axes[1].set_title('Negative Reviews Word Cloud', fontsize=16, fontweight='bold', color='red')
axes[1].axis('off')

plt.tight_layout()
plt.show()


# <h3>Export Updated Dataset with Sentiment Features</h3>




# Save updated dataframe with sentiment features back to processed data folder
output_path = '../data/processed/cleaned_reviews_sentiment.csv'
df_nlp.to_csv(output_path, index=False)
print(f"SUCCESS: Saved enriched sentiment dataset to {output_path}")







