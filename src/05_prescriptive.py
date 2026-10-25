#!/usr/bin/env python
# coding: utf-8

# # Prescriptive Analytics

# <h3>Imports & Setup</h3>




import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("Setup complete for Prescriptive Analytics!")


# <h3>Load Enriched Dataset</h3>




data_path = '../data/processed/cleaned_reviews_sentiment.csv'
if not os.path.exists(data_path):
    data_path = '../data/processed/cleaned_reviews.csv'

df = pd.read_csv(data_path, dtype={'product_name': str})
print(f"Loaded dataset for Prescriptive Analytics. Rows: {len(df):,}, Columns: {df.shape[1]}")


# <h3>Train Prescriptive Decision Tree Model</h3>




# Select key numeric, engagement, and sentiment features for prescriptive rules
feature_cols = ['review_char_count', 'review_word_count', 'has_title', 'helpful_votes']
if 'polarity' in df.columns:
    feature_cols.extend(['polarity', 'subjectivity'])

X = df[feature_cols].fillna(0)
y = df['recommended_ind'].values

# Fit a shallow Decision Tree (max_depth=3) for clear executive rule extraction
dt_model = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
dt_model.fit(X, y)

# Save model object
os.makedirs('../models', exist_ok=True)
joblib.dump(dt_model, '../models/decision_tree_prescriptive.pkl')

print(f"Prescriptive Decision Tree Trained! Model Accuracy: {accuracy_score(y, dt_model.predict(X)):.4f}\n")
print("==========================================================================")
print("                       PRESCRIPTIVE DECISION RULES                        ")
print("==========================================================================")
print(export_text(dt_model, feature_names=feature_cols))


# <h3>Visualizing the Decision Rule Tree</h3>




plt.figure(figsize=(18, 8))
plot_tree(
    dt_model,
    feature_names=feature_cols,
    class_names=['Not Recommended', 'Recommended'],
    filled=True,
    rounded=True,
    fontsize=11
)
plt.title("Prescriptive Decision Tree: Executive Root Cause Rules", fontsize=14, fontweight='bold')
plt.show()


# <h3>Gini Feature Importance Ranking (Key Drivers)</h3>




importances = dt_model.feature_importances_
feat_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=feat_importance, x='Importance', y='Feature',hue='Importance', palette='crest')
plt.title('Key Business Drivers of Customer Recommendation', fontsize=14, fontweight='bold')
plt.xlabel('Gini Importance Score')
plt.ylabel('Feature')
plt.show()

print("Feature Importance Breakdown:")
display(feat_importance)





