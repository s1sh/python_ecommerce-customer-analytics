"""
Script: src/01_data_prep.py
Purpose: Data Cleaning, Missing Value Substitution, and Feature Engineering
Run command: python src/01_data_prep.py

"""
#!/usr/bin/env python
# coding: utf-8

# # Data Prepartion 
# 

# <h3>importing pakages</h3>

# In[1]:


# importing pakages 
import os
import glob
import pandas as pd
import numpy as np


# <h3>Load Raw Data</h3>




# path of data
raw_data_path = '../data/raw/1429_1.csv'

# Check if file exists
if not os.path.exists(raw_data_path):
    raise FileNotFoundError(f"Raw dataset not found at {raw_data_path}. Please download and place it in data/raw/")

# Load raw dataset
df_raw = pd.read_csv(raw_data_path)
print(f"Raw Data Loaded Successfully! Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")


# <h3>Standardize Column Names</h3>




# Convert column names to snake_case

df_clean = df_raw.copy()

# Map columns standard names
column_mapping = {
    'id': 'product_id',
    'name': 'product_name',
    'brand': 'brand',
    'categories': 'category',
    'reviews.rating': 'rating',
    'reviews.text': 'review_text',
    'reviews.title': 'title',
    'reviews.doRecommend': 'recommended_raw',
    'reviews.numHelpful': 'helpful_votes'
}

df_clean = df_clean.rename(columns=column_mapping)

# Retain required fields
required_cols = ['product_id', 'product_name', 'brand', 'category', 'rating', 'review_text', 'title', 'recommended_raw', 'helpful_votes']
df_clean = df_clean[[col for col in required_cols if col in df_clean.columns]]

print("Standardized Column Names:")
print(list(df_clean.columns))


# <h3>Handle Missing Values</h3>




print("Missing values before cleaning:")
print(df_clean.isnull().sum())
print(f"\nTotal rows: {len(df_clean):,}")





# Substitute missing text fields instead of dropping records
df_clean['review_text'] = df_clean['review_text'].fillna('Missing')
df_clean['title'] = df_clean['title'].fillna('No Title')
df_clean['brand'] = df_clean['brand'].fillna('Amazon')
df_clean['category'] = df_clean['category'].fillna('Electronics')
df_clean['helpful_votes'] = df_clean['helpful_votes'].fillna(0)

# Ensure numeric ratings are clean
df_clean = df_clean.dropna(subset=['rating'])
df_clean['rating'] = df_clean['rating'].astype(int)

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())
print(f"\nTotal retained rows: {len(df_clean):,}")


# <h3>Variable Derivation</h3>




# Extract primary high-level category
df_clean['category'] = df_clean['category'].apply(
    lambda x: str(x).split(',')[0].strip() if pd.notnull(x) else 'Electronics'
)





print(df_clean['recommended_raw'].unique())
print(df_clean['rating'].unique())




# Binary Variable (recommended_indicator: 1 = Recommended, 0 = Not Recommended)

# Convert string boolean variations ('true'/'false') to 1/0
raw_map = {'true': 1, 'false': 0}
df_clean['recommended_ind'] = df_clean['recommended_raw'].astype(str).str.lower().map(raw_map)

# Impute missing values based on rating (Rating >= 4 is Recommended)
rating_good = (df_clean['rating'] >= 4).astype(int)
df_clean['recommended_ind'] = df_clean['recommended_ind'].fillna(rating_good).astype(int)


# String label for visualizations
df_clean['recommended_label'] = df_clean['recommended_ind'].map({1: 'Recommended', 0: 'Not Recommended'})

# 2. Text Metrics 
df_clean['review_char_count'] = df_clean['review_text'].apply(lambda x: 0 if x == 'Missing' else len(str(x)))
df_clean['review_word_count'] = df_clean['review_text'].apply(lambda x: 0 if x == 'Missing' else len(str(x).split()))
df_clean['has_title'] = df_clean['title'].apply(lambda x: 0 if x == 'No Title' else 1)

print("Variable Derivation Sample:")
df_clean[['review_text', 'rating', 'recommended_ind','recommended_label', 'review_word_count','review_char_count']].head()


# <h3>Export Cleaned Dataset</h3>




os.makedirs('../data/processed', exist_ok=True)
output_path = '../data/processed/cleaned_reviews.csv'

df_clean.to_csv(output_path, index=False)
print(f"SUCCESS: Amazon Consumer Reviews dataset processed and saved to {output_path}")
print(f"Final Processed Shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns.")







