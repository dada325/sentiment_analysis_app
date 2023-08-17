import streamlit as st
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

MODEL_DIR = "./saved_model/"
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)

st.title("Sentiment Analysis")

text = st.text_area("Enter Text:", "")
if st.button("Predict"):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    label = probs.argmax()
    st.write(f"Predicted Label: {label}, Confidence: {probs[label]:.2f}")
