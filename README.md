# 📧 Spam Classifier Using Naïve Bayes

## 📌 Project Overview
This project is a **Spam Classification Model** that uses **Naïve Bayes** to detect whether a message is **Spam** 🚨 or **Not Spam (Ham)** ✅. It processes messages using **Natural Language Processing (NLP)** and converts them into numerical features for classification.

## 🚀 Features
- 📜 Uses the **SMS Spam Collection Dataset** 📂
- 🛠 **Text Preprocessing**: Lowercasing, removing stopwords, tokenization, lemmatization, and cleaning text
- 🔢 **Feature Extraction**: Converts text into numerical features using **TF-IDF**
- 🤖 **Machine Learning Model**: Trains a **Multinomial Naïve Bayes model**
- 📊 **Performance Evaluation**: Calculates **Accuracy, Precision, Recall, and F1-score**
- 🎯 **Model Accuracy**: **97.31%** ✅

## 📂 Dataset
- **Dataset Name**: `spam.csv`
- **Source**: Available from **Kaggle** 📌
- **Columns**:
  - `v1`: Label (**Spam / Ham**)
  - `v2`: Message Text

## 🛠️ Tech Stack
- 🐍 **Python**
- 🤖 **Scikit-learn**
- 📝 **NLTK (Natural Language Toolkit)**
- 📊 **Pandas & NumPy**

## 📥 Installation & Setup
Follow these simple steps to set up and run the project on your system.

### 1️⃣ **Clone the Repository**
```bash
 git clone https://github.com/your-username/spam-classifier-naive-bayes.git
 cd spam-classifier-naive-bayes
```

### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3️⃣ **Run the Project**
```bash
python spam_classifier.py
```

## 📌 How It Works (Step-by-Step)
1️⃣ **Data Loading & Preprocessing** 📂
   - Loads the dataset (`spam.csv`)
   - Keeps only required columns (`v1` -> `label`, `v2` -> `message`)
   - Converts labels into binary (`ham = 0`, `spam = 1`)
   - Cleans text: Lowercasing, removing special characters, tokenization, removing stopwords, and lemmatization
   - Saves the cleaned dataset as `preprocessed_spam.csv`

2️⃣ **Feature Extraction (TF-IDF)** 🔢
   - Uses **TF-IDF Vectorizer** to convert text into numerical form
   - Selects **top 5000 important words**
   - Splits data into **Training (80%) & Testing (20%)**
   - Saves vectorizer as `tfidf_vectorizer.pkl`
   - Saves training/testing data as `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

3️⃣ **Training Naïve Bayes Model** 🤖
   - Uses **Multinomial Naïve Bayes Classifier**
   - Trains the model on `X_train` and `y_train`
   - Evaluates the model using `X_test` and `y_test`
   - Prints accuracy, precision, recall, and F1-score
   - Saves the trained model as `spam_classifier.pkl`

4️⃣ **Real-Time Message Prediction** 🎯
   - Loads the saved model and vectorizer
   - Takes user input (message)
   - Predicts whether the message is **Spam 🚨 or Not Spam ✅**

## 📊 Model Performance
✅ **Accuracy**: **97.31%**
📊 **Classification Report**:
```
              precision    recall  f1-score   support

           0       0.97      1.00      0.98       965
           1       1.00      0.80      0.89       150

    accuracy                           0.97      1115
   macro avg       0.98      0.90      0.94      1115
weighted avg       0.97      0.97      0.97      1115
```

## 📜 File Structure
```
spam-classifier-naive-bayes/
├── spam.csv                   # Original dataset
├── preprocessed_spam.csv       # Cleaned dataset
├── tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer
├── spam_classifier.pkl         # Trained model
├── X_train.csv / X_test.csv    # Training & testing data
├── y_train.csv / y_test.csv    # Labels
├── spam_classifier.py          # Main Python script
├── requirements.txt            # Required dependencies
└── README.md                   # This file
```

## 🎯 Future Improvements
✅ Improve model accuracy using **Deep Learning (LSTM, Transformers)**
✅ Add a **GUI/Web App** for easy access
✅ Deploy the model as a **REST API**

## 🙌 Conclusion
This project successfully implements a **Spam Classifier** using **Naïve Bayes** with an accuracy of **97.31%**! 🎯 You can now detect **spam messages in real-time** with high precision! 🚀

📌 **Try running the project & test it with real messages!** 📩

