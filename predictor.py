# predictor.py
# This file contains the logic for making predictions on new news texts.

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and stop words globally for efficiency (local to this file)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text_single(text):
    """
    This function cleans up a single piece of news text for predictor.py's use.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()

    cleaned_words = [
        lemmatizer.lemmatize(word) for word in words if word not in stop_words
    ]

    return ' '.join(cleaned_words)


def predict_news(text, tfidf_vectorizer, model):
    """
    This function takes a brand new piece of news and tells us if it's real or fake.
    """
    print(f"\n--- Predicting for new text ---")
    print(f"Original text: '{text[:100]}...'")

    # 1. Clean the new text, just like we cleaned the training data
    cleaned_text = clean_text_single(text) # Calls the function defined within this file
    print(f"Cleaned text: '{cleaned_text[:100]}...'")

    # 2. Convert the cleaned text into numbers using the SAME TF-IDF tool
    text_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # 3. Make a prediction using our trained model
    prediction = model.predict(text_vectorized)

    # The prediction will be a list, so we take the first (and only) item
    result = prediction[0]
    print(f"Prediction: This news is likely '{result}'")
    return result

