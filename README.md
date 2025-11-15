📊 Twitter Sentiment Analysis

Machine Learning | TF-IDF | Scikit-learn | Sentiment140 Dataset

📌 Project Overview

This project performs sentiment analysis on tweets using the Sentiment140 dataset (1.6 million tweets).
The goal is to classify tweets as positive or negative using traditional machine-learning models instead of deep learning.

The system uses:

TF-IDF vectorization for feature extraction

Bernoulli Naive Bayes, Linear SVM, and Logistic Regression for classification

Evaluation metrics including accuracy and classification reports

A simple custom tweet testing module at the end

🗂 Dataset

Sentiment140 Dataset

Contains 1.6M tweets

Labels:

0 → Negative

4 → Positive

Columns used:

polarity (target)

text (tweet content)

Dataset file used:

training.1600000.processed.noemoticon.csv.zip

⚙️ Tech Stack

Python

Pandas

Scikit-learn

TF-IDF Vectorizer

Machine Learning Models:

Bernoulli Naive Bayes

Linear SVM

Logistic Regression

🔧 Project Workflow
1. Data Loading & Preprocessing

Loads the CSV dataset (compressed .zip)

Extracts required columns

Removes neutral tweets

Converts sentiment labels to binary

2. Feature Extraction

Applies TF-IDF vectorization

Removes stopwords

Generates a sparse numerical feature matrix

3. Model Training

Trains 3 machine-learning classifiers:

BernoulliNB

LinearSVC

Logistic Regression

4. Model Evaluation

Prints:

Accuracy for each model

Precision, Recall, F1-score (classification report)

5. Testing on Custom Tweets

Example:

["I love this!", "I hate that!", "It was okay, not great."]


Models predict the sentiment for each.

📈 Results

Typical findings:

SVM achieves the best accuracy

Logistic Regression performs competitively

BernoulliNB is fast but less precise

▶️ How to Run
1. Install dependencies
pip install pandas scikit-learn

2. Add the dataset

Place the training.1600000.processed.noemoticon.csv.zip file in the project directory.

3. Run the notebook
jupyter notebook


Open Source-code.ipynb.

📦 Project Structure
Twitter-Sentiment-Analysis/
│
├── Source-code.ipynb
├── training.1600000.processed.noemoticon.csv.zip
├── README.md
└── requirements.txt (optional)

📝 Future Improvements

✔ Add LSTM/BERT-based models
✔ Enhance text preprocessing
✔ Build a simple UI or API endpoint
✔ Add visualizations like sentiment distribution
