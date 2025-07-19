# main.py
# This is the main brain that runs our robot detective!

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib # Import joblib for saving/loading models

# Imports for other modules
from data_loader import load_dataset, get_data_for_training
from feature_extractor import extract_features, split_data
from model_trainer import train_model, evaluate_model
from predictor import predict_news # This will now use its own internal clean_text_single

# --- Text Preprocessing Functions (Self-contained within main.py) ---
# Initialize lemmatizer and stop words globally for efficiency in this file
lemmatizer_main = WordNetLemmatizer()
stop_words_main = set(stopwords.words('english'))

def clean_text_single_main(text):
    """
    This function cleans up a single piece of news text for main.py's use.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic/non-whitespace characters

    words = text.split()

    cleaned_words = [
        lemmatizer_main.lemmatize(word) for word in words if word not in stop_words_main
    ]

    return ' '.join(cleaned_words)

def preprocess_texts(texts):
    """
    This function applies the cleaning process to many news texts for main.py's use.
    """
    print("Starting preprocessing of text...")
    cleaned_texts_list = [clean_text_single_main(text) for text in texts]
    print("Finished preprocessing of text...")
    return cleaned_texts_list
# --- End Text Preprocessing Functions ---


def run_fake_news_detector():
    """
    This is the main function that runs our entire fake news detection project.
    It orchestrates all the steps: load, clean, convert, train, evaluate, and predict.
    """
    print("--- Starting Fake News Detector Project ---")

    # --- Step 0: Ensure NLTK data is downloaded (run once) ---
    print("Ensuring NLTK data (stopwords, wordnet) is present...")
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("NLTK stopwords and wordnet confirmed to be present.")
    except Exception as e:
        print(f"An error occurred during NLTK data download: {e}")
        print("Please ensure you have an active internet connection or try running 'python -m nltk.downloader all' in your terminal.")
        print("Exiting due to NLTK download issue.")
        return
    print("-" * 30)

    # --- Step 1: Load the Dataset (Robot's Practice Books) ---
    df = load_dataset()
    if df is None:
        print("Exiting due to data loading error.")
        return

    # Get the news texts (X) and their labels (y)
    X, y = get_data_for_training(df)
    if X is None or y is None:
        print("Exiting due to data preparation error.")
        return

    print("-" * 30)

    # --- Step 2: Preprocess Texts (Clean Up the Words) ---
    print(f"DEBUG: Type of X before preprocessing: {type(X)}")
    cleaned_texts = preprocess_texts(X)
    print(f"DEBUG: Type of cleaned_texts after preprocessing: {type(cleaned_texts)}")
    print("-" * 30)

    # --- Step 3: Extract Features (Turn Words into Numbers) ---
    tfidf_vectorizer, X_vectorized = extract_features(cleaned_texts)
    print("-" * 30)

    # --- Step 4: Split Data (Separate Practice Books from Test Books) ---
    X_train, X_test, y_train, y_test = split_data(X_vectorized, y)
    print("-" * 30)

    # --- Step 5: Train the Model (Teach Our Robot Detective) ---
    model = train_model(X_train, y_train)
    print("-" * 30)

    # --- Step 6: Evaluate the Model (See How Good Our Robot Is!) ---
    evaluate_model(model, X_test, y_test)
    print("-" * 30)

    # --- Step 7: Save the Trained Model and Vectorizer ---
    print("Saving trained model and TF-IDF vectorizer...")
    try:
        joblib.dump(model, 'fake_news_model.pkl')
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
        print("Model and vectorizer saved successfully as 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl'.")
    except Exception as e:
        print(f"Error saving model: {e}")
    print("-" * 30)

    # --- Step 8: Make Predictions on New News (Robot's First Real Cases!) ---
    print("Let's try predicting some new news articles!")

    sample_news_1 = "BREAKING NEWS! Aliens landed in New York and declared pizza the universal currency. Scientists baffled."
    predict_news(sample_news_1, tfidf_vectorizer, model)

    sample_news_2 = "The local city council approved a new budget for park maintenance, focusing on improving green spaces for residents."
    predict_news(sample_news_2, tfidf_vectorizer, model)

    sample_news_3 = "Exclusive: A new study reveals that eating chocolate every day makes you invisible. Doctors recommend a daily bar."
    predict_news(sample_news_3, tfidf_vectorizer, model)

    sample_news_4 = "President signs new bill into law aimed at reducing carbon emissions by 30% over the next decade. Environmental groups praise the move."
    predict_news(sample_news_4, tfidf_vectorizer, model)

    print("\n--- Fake News Detector Project Finished ---")

# This line makes sure our function runs when we start the main.py file
if __name__ == "__main__":
    run_fake_news_detector()

