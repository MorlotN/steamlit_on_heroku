import tensorflow as tf
import pickle
import streamlit as st
from keras.preprocessing import sequence

with open('heroku_app/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
txt = st.text_area('Text to analyze', 'this is a bad comments very bad')
data = sequence.pad_sequences(tokenizer.texts_to_sequences([txt]), maxlen=100)


model = tf.keras.models.load_model("heroku_app/simple")


y_pred = model.predict(x=[data])
pred_labels = y_pred.round()
if pred_labels==0:
    res = "bad comment"
else:
    res = "good comment"

st.write(res)
