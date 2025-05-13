import streamlit as st
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Download resources NLTK 
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except nltk.downloader.DownloadError:
    nltk.download('omw-1.4')

# Inisialisasi lemmatizer dan stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_model_and_vectorizer():
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    with open('LR_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

def clean_predict_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        tokens = word_tokenize(text)
        filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w.isalpha()]
        return ' '.join(filtered_tokens)
    return ''

def predict_fake_news(news_text, model, vectorizer):
    cleaned_text = clean_predict_text(news_text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction_probability = model.predict_proba(text_vectorized)[:, 1] # Probabilitas kelas 'REAL' (1)
    prediction_label = "REAL" if prediction_probability > 0.5 else "FAKE"
    confidence = prediction_probability if prediction_label == "REAL" else 1 - prediction_probability
    return prediction_label, confidence

st.title("Fake News Prediction")
st.subheader("Input news text here:")

news_input = st.text_area("News Text", height=200)

if st.button("Predict"):
    if news_input:
        prediction, confidence = predict_fake_news(news_input, model, vectorizer)
        st.subheader("Result:")
        if prediction == "REAL":
            st.success(f"This news is predicted to be: {prediction} (Confidence: {confidence:.4f})")
        else:
            st.error(f"This news is predicted to be: {prediction} (Confidence: {confidence:.4f})")
    else:
        st.warning("Please input news text.")