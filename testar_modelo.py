import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('modelo_sonolencia.keras')

def prever(caminho):
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (96, 96))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 96, 96, 1)
    return model.predict(img, verbose=0)[0][0]

for caminho in ['teste_Closed_Eyes.jpg', 'teste_Open_Eyes.jpg']:
    p = prever(caminho)
    print(f'{caminho}: prob. de aberto = {p:.3f}')