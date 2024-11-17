from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import gradio as gr
model = load_model('TrafficSign.h5')
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris si nécessaire
    img = img / 255.0  # Normaliser les valeurs entre 0 et 1
    return img

def predict_image_class(image):

    img = np.array(image)
    # Prétraitement de l'image
    img = cv2.resize(img, (32, 32))  # Redimensionner l'image
    img = preprocessing(img)         # Prétraitement supplémentaire si nécessaire
    img = img.reshape(1, 32, 32, 1)  # Adapter la forme de l'image pour le modèle

    # Prédire la classe avec le modèle
    predicted_probabilities = model.predict(img)
    predicted_class = int(np.argmax(predicted_probabilities))

    return str(predicted_class)

# Interface Gradio
interface = gr.Interface(
    fn=predict_image_class,  # Fonction de classification
    inputs=gr.Image(type="pil"),  # Entrée : Image téléchargée
    outputs=gr.Textbox(label="Résultat de la Classification"),  # Sortie : Texte
    title="Classification d'Images",
    description="Téléchargez une image pour obtenir sa classe prédite."
)

# Lancer l'interface
if __name__ == "__main__":
    interface.launch()