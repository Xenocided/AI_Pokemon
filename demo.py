import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image as tf_image


st.title('Pokemon classification')

model = load_model('model.hdf5')

img_file = st.file_uploader('Please upload an image.')

IMG_SIZE_HEIGHT = 96
IMG_SIZE_WIDTH = 96

def load_result(img_file, predicted_index):
	label_map = ['bulbasaur','charmander','mewtwo','pikachu','squirtle']
	st.info(f'The image is a {label_map[predicted_index] }')
	st.image(img_file,width=400)

if img_file is not None:

	img = Image.open(img_file).convert('RGB')
	img = np.asarray(img)
	img = tf.image.resize(img, size=(IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH))
	img = img/255
	img = np.expand_dims(img, axis=0)

	predictions = model.predict(img)
	predicted_index = np.argmax(predictions, axis=1)[0]
	load_result(img_file, predicted_index)
	





	
