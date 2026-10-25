================================================================================

E-commerce customer analytics(consumer-reviews-of-amazon-products kaggel-dataset)

Course: Business Intelligence and Analytics (MBAAN-211)

Dataset: Datafiniti Consumer Reviews of Amazon Products (1429_1.csv)

================================================================================




PROJECT OVERVIEW

--------------------------------------------------------------------------------

This project delivers an end-to-end data analytics and predictive machine

learning pipeline built on ~34,000 Amazon consumer electronics reviews. 

The system analyzes key performance indicators (KPIs), extracts customer sentiment

using Natural Language Processing (NLP), benchmarks predictive machine learning 

classifiers (KNN, SVM, MLP Neural Networks), and derives prescriptive decision

rules using Decision Trees.





HOW TO VIEW NOTEBOOK OUTPUTS IN BROWSER WITHOUT JUPYTERLAB OR PYTHON (HTML REPORTS)

--------------------------------------------------------------------------------

If you do not have JupyterLab or Python installed, open any of the pre-rendered

HTML files in the "reports/html_notebooks/" directory using standard web browsers

(Chrome, Edge, Firefox, Safari):



- reports/html_notebooks/01_data_prep.html

- reports/html_notebooks/02_descriptive.html

- reports/html_notebooks/03_text_mining.html

- reports/html_notebooks/04_predictive.html

- reports/html_notebooks/05_prescriptive.html





HOW TO RUN THE PROJECT WITHOUT JUPYTERLAB (COMMAND LINE / STANDALONE)

--------------------------------------------------------------------------------

Anyone can execute the complete end-to-end pipeline using Python scripts:



1. Activate your virtual environment (optional but recommended):

   - Windows PowerShell: .\venv\Scripts\Activate.ps1

   - Linux/Mac: source venv/bin/activate

2 Install Python library dependencies given in file requirements.txt 
   
   pip install -r requirements.txt


3. Run the scripts sequentially:

   python src/01_data_prep.py

   python src/02_descriptive.py

   python src/03_text_mining.py

   python src/04_predictive.py

   python src/05_prescriptive.py




DIRECTORY STRUCTURE SUMMARY

--------------------------------------------------------------------------------

python_consumer-reviews-analytics/

├── README.txt                       <-- Plain text documentation (Notepad friendly)

├── PROJECT_STRUCTURE.txt            <-- Complete directory tree guide

├── requirements.txt                 <-- Python library dependencies

├── data/

│   ├── raw/                         <-- Raw Kaggle dataset (1429_1.csv)

│   └── processed/                   <-- Cleaned datasets (cleaned_reviews_sentiment.csv)

├── notebooks/                       <-- Interactive Jupyter Notebooks (.ipynb)

├── src/                             <-- Executable Standalone Python Scripts (.py)

├── models/                          <-- Saved Trained Model Binaries (.pkl)

└── reports/

    └── html_notebooks/              <-- Exported HTML Notebooks (Browser readable)





AUTHOR & CONTACT

--------------------------------------------------------------------------------

Course Project: MBAAN-211 Business Intelligence & Analytics

Tech Stack: Python,jupyterlab, Pandas, Scikit-Learn, NLTK, TextBlob

================================================================================