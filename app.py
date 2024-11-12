import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Charger le modèle .h5
model = load_model('TrafficSign.h5')

# Fonction de prétraitement (à adapter selon les besoins du modèle)
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris si nécessaire
    img = img / 255.0  # Normaliser les valeurs entre 0 et 1
    return img

# Charger et traiter une image
def predict_image_class(image_path):
    img = cv2.imread(image_path)  # Lire l'image depuis le chemin
    img = np.asarray(img)  # Convertir en tableau numpy
    img = cv2.resize(img, (32, 32))  # Redimensionner l'image à 32x32
    img = preprocessing(img)  # Prétraiter l'image
    #plt.imshow(img, cmap=plt.get_cmap('gray'))  # Afficher l'image prétraitée
    #plt.show()
    
    # Changer la forme de l'image pour correspondre à l'entrée du modèle
    img = img.reshape(1, 32, 32, 1)  # Ajouter des dimensions pour correspondre au modèle
    
    # Prédire la probabilité pour chaque classe
    predicted_probabilities = model.predict(img)
    
    # Trouver la classe avec la probabilité la plus élevée
    predicted_class = int(np.argmax(predicted_probabilities))
    
    print("Classe prédite:", predicted_class)

# Exemple d'utilisation
predict_image_class('stop.jpg')
