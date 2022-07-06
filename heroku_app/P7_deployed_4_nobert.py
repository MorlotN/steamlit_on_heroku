# Commented out IPython magic to ensure Python compatibility.
# import pandas as pd
import tensorflow as tf
# import numpy as np
import pickle
import streamlit as st
# from tensorflow import keras
from keras.preprocessing import sequence

# loading
with open('heroku_app/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
txt = st.text_area('Text to analyze', 'this is a bad comments very bad')
data = sequence.pad_sequences(tokenizer.texts_to_sequences([txt]), maxlen=100)


model = tf.keras.models.load_model("heroku_app/simple")


y_pred = model.predict(x=[data])
pred_labels = y_pred.round()
if pred_labels==0:
    # print("bad comment")
    res = "bad comment"
else:
    # print("good comment")
    res = "good comment"

# print(pred_labels)

st.write(res)
st.write(pred_labels)