import streamlit as st
import sklearn
import helper
import pickle
import nltk
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image  

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

model=pickle.load(open("models/model.pkl",'rb'))

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background: rgb(173, 216, 230);
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
class_names = ['angry', 'fear', 'happy', 'neutral', 'sad','surbrise']

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title('Facial Expressions App')
st.text('Hello, world!')

img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if img is not None:
    img = Image.open(img)
    st.image(img, caption="Uploaded Image")

    img = np.array(img)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img,(48,48))
    img = img/255.0
    img = np.expand_dims(img,axis=0)
    pred = model.predict(img)
    number_of_class_float = np.argmax(pred)
    number_of_class_int = int(number_of_class_float)

state = st.button('Predict')
if state:
    st.markdown((
    f"Expression : <u><b>{class_names[number_of_class_int]} </b></u>"
), unsafe_allow_html=True)
