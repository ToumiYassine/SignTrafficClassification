import os
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
from pydantic import BaseModel

# Créer une instance de FastAPI
app = FastAPI()

# Charger le modèle .h5 pour la classification
model = load_model('TrafficSign.h5')

# Dossier où les fichiers uploadés seront stockés
UPLOAD_DIR = "uploaded_images"

# Créer le dossier si ce n'est pas déjà fait
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Fonction de prétraitement de l'image
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris si nécessaire
    img = img / 255.0  # Normaliser les valeurs entre 0 et 1
    return img

# Modèle pour accepter les données JSON pour le nom du fichier
class FileName(BaseModel):
    file_name: str

# Fonction pour prédire la classe de l'image
def predict_image_class(file_path: str):
    # Essayer de lire et traiter l'image
    try:
        image = Image.open(file_path).convert("RGB")  # Assurer la conversion en RGB
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File could not be read as an image: {str(e)}")

    img = np.array(image)

    # Prétraitement de l'image
    img = cv2.resize(img, (32, 32))  # Redimensionner l'image
    img = preprocessing(img)         # Prétraitement supplémentaire si nécessaire
    img = img.reshape(1, 32, 32, 1)  # Adapter la forme de l'image pour le modèle

    # Prédire la classe avec le modèle
    predicted_probabilities = model.predict(img)
    predicted_class = int(np.argmax(predicted_probabilities))

    return str(predicted_class)

# Endpoint combiné pour uploader et prédire l'image
@app.post("/upload_and_predict")
async def upload_and_predict(file: UploadFile = File(...)):
    try:
        # Lire le contenu du fichier uploadé
        contents = await file.read()

        # Déterminer le chemin du fichier à sauvegarder
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Sauvegarder le fichier sur le serveur
        with open(file_path, 'wb') as f:
            f.write(contents)

        # Prédire la classe de l'image après l'avoir téléchargée
        predicted_class = predict_image_class(file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")
    finally:
        file.file.close()

    return {"predicted_class": predicted_class}
