from flask import Flask, render_template, request, jsonify
import re
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Charge ton modèle (assure-toi d'avoir un fichier model.h5 dans le dossier)
model = load_model('model.h5')

def preprocess_image(base64_str):
    # Nettoyage de la chaîne base64 et conversion en image
    image_data = re.sub('^data:image/.+;base64,', '', base64_str)
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    
    # Conversion en niveaux de gris et redimensionnement strict 28x28
    img = img.convert('L').resize((28, 28))
    
    # Normalisation pour le modèle (0 à 1)
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict(): # Inference
    data = request.get_json() # Get the image to predict
    img_array = preprocess_image(data['image'])
    
    # Simulation de prédiction si le modèle n'est pas chargé (pour test)
    prediction = model.predict(img_array)[0]
    #prediction = np.random.dirichlet(np.ones(10), size=1)[0] # À remplacer par model.predict
    
    return jsonify({
        'digit': int(np.argmax(prediction)),
        'confidence': float(np.max(prediction)),
        'probabilities': prediction.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
