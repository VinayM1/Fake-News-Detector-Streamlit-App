# app.py
import streamlit as st
import joblib # For loading the saved model and vectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os # To check if model files exist

# --- NLTK Data Download (for Streamlit app) ---
@st.cache_resource # Cache NLTK downloads to avoid repeated operations
def download_nltk_data():
    """Ensures NLTK data is downloaded for the Streamlit environment."""
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/stopwords')
        # st.success("NLTK data (wordnet, stopwords) confirmed present.") # Can uncomment for verbose output
    except Exception: # Catch any exception, including DownloadError if it exists or not
        st.warning("⏳ NLTK data (wordnet, stopwords) not found. Attempting download...")
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        st.success("✅ NLTK data download complete.")

download_nltk_data()


# --- Text Preprocessing Functions (Self-contained within app.py) ---
# Initialize lemmatizer and stop words globally for efficiency in this file
lemmatizer_app = WordNetLemmatizer()
stop_words_app = set(stopwords.words('english'))

def clean_text_single_app(text):
    """
    This function cleans up a single piece of news text for app.py's use.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic/non-whitespace characters

    words = text.split()

    cleaned_words = [
        lemmatizer_app.lemmatize(word) for word in words if word not in stop_words_app
    ]

    return ' '.join(cleaned_words)
# --- End Text Preprocessing Functions ---

# --- Load Model ---
# Define model file paths
MODEL_PATH = 'fake_news_model.pkl'
TFIDF_PATH = 'tfidf_vectorizer.pkl'

# Check if model files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
    st.error("❌ Model files not found! Please run `main.py` first to train and save the model.")
    st.info("Make sure 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as 'app.py'.")
    st.stop() # Stop the Streamlit app if models are not found

@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_ml_resources():
    """Loads the pre-trained model and TF-IDF vectorizer."""
    try:
        model = joblib.load(MODEL_PATH)
        tfidf_vectorizer = joblib.load(TFIDF_PATH)
        return model, tfidf_vectorizer
    except Exception as e:
        st.error(f"Failed to load model resources: {e}")
        st.stop()

model, tfidf_vectorizer = load_ml_resources()

# --- Streamlit UI ---
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered", # Can be "wide" for more space
    initial_sidebar_state="collapsed" # "expanded" or "collapsed"
)

# Custom header with emoji
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
        <span style="font-size: 50px;">📰</span>
        <h1 style="color: #6495ED; text-align: center;">Fake News Detector</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    **Unsure if that headline is real?** Paste a news article or headline below, and our Machine Learning model will predict if it's likely **REAL** or **FAKE**.
""")

# Added a prominent disclaimer about model limitations
st.info("""
    **💡 Important Note:** This model classifies news based on **linguistic patterns** learned from its training data, not real-time factual knowledge or current events. Therefore, a factually correct statement might still be classified as 'FAKE' if its phrasing or word usage aligns with patterns found in the fake news examples the model was trained on.
""")

st.markdown("---") # Visual separator

# Main input section
st.subheader("📝 Enter News Article Here:")
news_input = st.text_area(" ", height=250, placeholder="E.g., 'Scientists discover evidence of alien life on Mars, shocking the world.'", key="news_text_area") # Added a key for unique widget

col1, col2 = st.columns([1, 4]) # Use columns for button alignment
with col1:
    predict_button = st.button("Predict", type="primary", use_container_width=True) # Make button primary and full width

if predict_button:
    if news_input:
        with st.spinner("Analyzing news text..."):
            # 1. Clean the input text
            cleaned_input = clean_text_single_app(news_input)

            # 2. Vectorize the cleaned text using the loaded TF-IDF vectorizer
            input_vectorized = tfidf_vectorizer.transform([cleaned_input])

            # 3. Make prediction
            prediction = model.predict(input_vectorized)[0] # Get the single prediction result

            st.markdown("### Prediction Result:")
            if prediction == 'FAKE':
                st.error("🚨 This news is likely **FAKE**")
                st.balloons() # Visual celebration for fake news (ironic, but fun)
            else:
                st.success("✅ This news is likely **REAL**")
                st.snow() # Visual celebration for real news

            st.markdown("---")
            st.markdown("#### Input Details:")
            st.expander("View Original Input & Cleaned Text").markdown(f"""
                **Original Input:**
                ```
                {news_input[:500]}{'...' if len(news_input) > 500 else ''}
                ```

                **Cleaned Text:**
                ```
                {cleaned_input[:500]}{'...' if len(cleaned_input) > 500 else ''}
                ```
            """)

    else:
        st.warning("☝️ Please enter some text into the box to get a prediction.")

st.markdown("---") # Another visual separator

# Collapsible "How it Works" section
with st.expander("📚 How It Works: The Machine Learning Behind It"):
    st.markdown("""
    This application leverages a supervised Machine Learning pipeline to classify news articles:
    """)
    st.markdown("""
    1.  **Data:** The model was trained on a comprehensive dataset of real and fake news articles,
        including content from sources like `Fake.csv` and `True.csv`.
    2.  **Preprocessing:** Raw text is meticulously cleaned (lowercased, punctuation removed,
        stop words filtered, and lemmatized) to prepare it for analysis.
    3.  **Feature Extraction:** Cleaned text is converted into numerical features using
        **TF-IDF (Term Frequency-Inverse Document Frequency)**, highlighting words
        important to specific articles.
    4.  **Machine Learning Model:** A **Passive Aggressive Classifier** learns patterns from
        these numerical features to distinguish between **REAL** and **FAKE** news.
    """)

st.markdown("---") # Final visual separator

st.markdown("""
    **Connect with me:** https://www.linkedin.com/in/vinay-meena-036119326/ | **GitHub Repository:** https://github.com/VinayM1
""")
