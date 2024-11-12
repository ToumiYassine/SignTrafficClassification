import requests

# URL de l'API (ajuster selon l'adresse de ton serveur local ou de production)
url = "http://127.0.0.1:8000/upload_and_predict"

# Chemin de l'image à tester
image_path = "30km.png"

# Ouvrir le fichier image en mode binaire
with open(image_path, "rb") as image_file:
    # Envoi de la requête POST avec l'image en tant que fichier
    files = {"file": ("image.jpg", image_file, "image/jpeg")}
    response = requests.post(url, files=files)

# Afficher la réponse de l'API
if response.status_code == 200:
    print("Réponse du serveur : ", response.json())
else:
    print(f"Erreur {response.status_code}: {response.text}")
