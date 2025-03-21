import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np

# Load the saved model and tokenizer
model_directory = "bert_sentiment_model"
tokenizer = BertTokenizer.from_pretrained(model_directory)
model = TFBertForSequenceClassification.from_pretrained(model_directory)

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=200)
    logits = model(**inputs).logits
    prediction = np.argmax(logits, axis=1).item()
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment

# Streamlit UI
st.title("Real-Time Sentiment Analysis")
st.write("Enter a review or feedback to analyze sentiment.")

# Text input box
user_input = st.text_area("Enter text:", "")

if st.button("Predict"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"### Sentiment: {sentiment}")
    else:
        st.write(f"### Sentiment:positive.")
