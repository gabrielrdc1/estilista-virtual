import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image


MODEL_PATH = os.path.join("modelo", "cnn_model_mobilenetv2.h5")

CLASS_NAMES = [
    'blusa_feminino', 'blusa_masculino', 'calca_feminino', 'calca_masculino', 'tenis_feminino', 'tenis_masculino'
]

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img_path, target_size=(180, 180)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classificar_imagem(img_path):
    img_pre = preprocess_image(img_path)
    pred = model.predict(img_pre)
    classe_idx = np.argmax(pred)
    confianca = float(np.max(pred))
    return CLASS_NAMES[classe_idx], confianca

