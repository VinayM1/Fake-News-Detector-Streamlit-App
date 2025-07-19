# feature_extractor.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def extract_features(texts):
    """
    This function turns our cleaned words into numbers that the robot can understand.
    It uses TF-IDF, which gives higher scores to important words in each story.

    Args:
        texts (list): A list of cleaned news texts.

    Returns:
        tuple: (tfidf_vectorizer, X_vectorized)
               tfidf_vectorizer: The trained TF-IDF tool (needed later for new news).
               X_vectorized: The news texts converted into a numerical format.
    """
    print("Starting feature extraction (Turning Words into Numbers)...")

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = tfidf_vectorizer.fit_transform(texts)

    print("Feature Extraction Completed. Texts Converted to Numerical Vectors")
    return tfidf_vectorizer, X_vectorized

def split_data(X, y , test_size=0.2, random_state=42):
    """
    This function splits our data into two groups:
    One group for the robot to learn from (training data).
    Another group for us to test the robot on (test data) - stories it hasn't seen.

    Args:
        X (sparse matrix): The numerical features of the news texts.
        y (pandas.Series): The labels ('REAL' or 'FAKE').
        test_size (float): What percentage of data to use for testing (e.g., 0.2 means 20%).
        random_state (int): A number to make sure our split is the same every time we run it.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"Splitting Data into Training and Testing Sets (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print("Data split Complete.")

    print(f"Training Set Complete: {X_train.shape[0]} samples")
    print(f"Test Set Complete: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

