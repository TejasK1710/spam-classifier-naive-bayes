# -*- coding: utf-8 -*-
"""spam-neive.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jR1KBJjbzpuFWUAMRB_SANX0UKaWQ6ft

** Spam Classifier Using Naïve Bayes **
"""

import pandas as pd
import numpy as np

df = pd.read_csv('spam.csv',encoding="latin-1")
df.head()

# Keep only necessary columns
df = df[['v1', 'v2']]  # Selecting only required columns

# Rename columns for better readability
df.columns = ['label', 'message']

# Convert labels to binary (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Save cleaned dataset
df.to_csv("cleaned_spam.csv", index=False)

print("✅ Data Preprocessing Complete. Cleaned file saved as cleaned_spam.csv")

df.head()

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load cleaned dataset
df = pd.read_csv("cleaned_spam.csv")

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    words = word_tokenize(text)  # Tokenization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords & lemmatize
    return " ".join(words)

# Apply preprocessing
df["processed_message"] = df["message"].apply(preprocess_text)

# Save preprocessed data
df.to_csv("preprocessed_spam.csv", index=False)

print("✅ Text Preprocessing Complete. Data saved as preprocessed_spam.csv")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# Load preprocessed dataset
df = pd.read_csv("preprocessed_spam.csv")

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Keep top 5000 important words

# Convert text into numerical features
df["processed_message"] = df["processed_message"].fillna("")  # Replace NaN with empty string
X = vectorizer.fit_transform(df["processed_message"]).toarray()

y = df["label"]  # Assuming 'label' column contains spam/ham (0 for ham, 1 for spam)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save TF-IDF model for future use
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save training & testing data
pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

print("✅ Feature extraction completed! TF-IDF model and datasets saved.")

import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the training and testing data
X_train = pd.read_csv("X_train.csv").values
X_test = pd.read_csv("X_test.csv").values
y_train = pd.read_csv("y_train.csv").values.ravel()  # Convert to 1D array
y_test = pd.read_csv("y_test.csv").values.ravel()

# Train Naïve Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the trained model
with open("spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model training complete! Saved as 'spam_classifier.pkl'.")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Print classification report
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and TF-IDF vectorizer
with open("spam_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to predict if a message is spam or not
def predict_message(message):
    # Convert the message into a TF-IDF feature vector
    message_tfidf = vectorizer.transform([message]).toarray()

    # Predict using the trained model
    prediction = model.predict(message_tfidf)[0]

    # Display result
    if prediction == 1:
        print("🚨 Spam Message!")
    else:
        print("✅ Not Spam (Ham)")

# Test the function
while True:
    user_message = input("\nEnter a message to check (or type 'exit' to quit): ")
    if user_message.lower() == "exit":
        break
    predict_message(user_message)