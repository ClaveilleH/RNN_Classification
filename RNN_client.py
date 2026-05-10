import requests
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json
import random

# URL = "http://127.0.0.1:5000"
URL = "http://rnn.dokpoly.claveille.fr"

class MNISTClient:
    def __init__(self, server_url=URL):
        self.url = f"{server_url}/predict"

    def array_to_base64(self, image_array):
        """Transforme un array NumPy (28,28) en chaîne Base64 compatible avec le serveur."""
        # Convertir l'array en objet Image PIL
        # On multiplie par 255 si les données sont normalisées entre 0 et 1
        img = Image.fromarray((image_array * 255).astype(np.uint8), mode='L')
        
        # Simuler un fichier PNG en mémoire
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        
        # Encoder en base64
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"

    def send_prediction(self, image_array, true_label):
        """Envoie l'image au serveur et compare le résultat."""
        b64_data = self.array_to_base64(image_array)
        
        payload = json.dumps({"image": b64_data})
        headers = {'Content-Type': 'application/json'}

        try:
            response = requests.post(self.url, data=payload, headers=headers)
            result = response.json()
            
            print(f"--- Test de prédiction ---")
            print(f"Chiffre attendu (Vrai) : {true_label}")
            print(f"Chiffre prédit par l'IA : {result['digit']}")
            
            # Affichage des probabilités (si disponibles)
            if 'probabilities' in result:
                conf = max(result['probabilities']) * 100
                print(f"Confiance : {conf:.2f}%")
            
            return result
        except Exception as e:
            print(f"Erreur lors de la requête : {e}")

# --- Script de test ---
if __name__ == "__main__":
    # 1. Charger le dataset MNIST (via Keras juste pour le test, ou un fichier .npy)
    # Si tu n'as pas keras, tu peux télécharger le fichier mnist.npz manuellement
    from tensorflow.keras.datasets import mnist
    (_, _), (x_test, y_test) = mnist.load_data()

    # client = MNISTClient("http://127.0.0.1:5000")
    client = MNISTClient(URL)

    # 2. Choisir 5 images au hasard et les envoyer
    for _ in range(50):
        index = random.randint(0, len(x_test) - 1)
        image = x_test[index] / 255.0  # Normalisation
        label = y_test[index]
        
        client.send_prediction(image, label)
        print("-" * 30)
