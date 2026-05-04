from flask import Flask, render_template, request, jsonify
import re
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
#from werkzeug.utils import secure_filename
import os


# Dossier où stocker les modèles uploadés
UPLOAD_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class DigitClassifier:
    """Gère uniquement le modèle et l'inférence """
    def __init__(self, model_path):
        self.model_path = model_path # tensorflows load_model => uses a file !
        self.load_model_from_path(model_path)

    def load_model_from_path(self, path):
        """
        Charge ou recharge le modèle.
        Il y a la structure ET les coefficients !!!
        """
        self.model = load_model(path)
        self.model_path = path
        
    def _preprocess(self, base64_str):
        # Nettoyage et conversion
        image_data = re.sub('^data:image/.+;base64,', '', base64_str) # on enleve les metadata placé en entete par la
                                                                                                               # fonction canvas.toDataURL() du front
        img = Image.open(BytesIO(base64.b64decode(image_data))) # le open => permet de gerer 
        """
        C'est la reconstruction de l'image en mémoire vive (RAM).
        * base64.b64decode(image_data) : Convertit le texte nettoyé en données binaires brutes (des octets).
        *BytesIO(...) : Simule un fichier. Au lieu d'enregistrer l'image sur ton disque dur pour la réouvrir,
          Python crée un "fichier virtuel" dans la mémoire vive. C'est beaucoup plus rapide.
        * Image.open(...) : La bibliothèque PIL (Pillow) lit ce fichier virtuel et identifie qu'il s'agit d'une image (PNG ou JPEG).
        """
        img = img.convert('L').resize((28, 28))  # Le mode 'L' (Luminance) transforme chaque pixel coloré en une valeur unique : niv de gris
        return np.array(img).reshape(1, 28, 28, 1) / 255.0

    def predict(self, base64_image):
        img_array = self._preprocess(base64_image)
        prediction = self.model.predict(img_array)[0]
        return {
            'digit': int(np.argmax(prediction)),
            'confidence': float(np.max(prediction)),
            'probabilities': prediction.tolist()
        }

    
class DigitServer:
    """Gère le serveur web et les routes."""
    def __init__(self, classifier: DigitClassifier):
        self.app = Flask(__name__)
        self.classifier = classifier
        self._register_routes()

    def _register_routes(self):
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/predict', 'predict', self.predict, methods=['POST'])
        self.app.add_url_rule('/upload', 'upload_model', self.upload_model, methods=['POST'])

    def index(self): # cf https://www.geeksforgeeks.org/python/flask-rendering-templates/
        return render_template('index.html')

    def predict(self):
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        result = self.classifier.predict(data['image'])
        return jsonify(result)

    def upload_model(self):
        """Reçoit le modèle dans la requête POST et le charge en mémoire."""
        if 'model' not in request.files:
            return jsonify({'error': 'Aucun fichier nommé "model" trouvé'}), 400
        
        file = request.files['model']
        
        if file.filename == '' or not file.filename.endswith('.keras'):
            return jsonify({'error': 'Format de fichier invalide (attendu .keras)'}), 400

        # Utilisation de tempfile pour manipuler le fichier en mémoire 
        # sans le stocker de façon permanente sur le disque
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=True) as tmp:
            file.save(tmp.name) # Enregistrement temporaire
            
            try:
                # On recharge le modèle à partir du chemin temporaire
                self.classifier.load_model_from_path(tmp.name)
                return jsonify({'message': 'Modèle mis à jour en mémoire avec succès'})
            except Exception as e:
                return jsonify({'error': f'Erreur lors du chargement : {str(e)}'}), 500

            
    def run(self, debug=True):
        self.app.run(debug=debug)

        
#===================================
if __name__ == '__main__':
    # Initialisation
    classifier = DigitClassifier('model.keras')
    server = DigitServer(classifier)
    
    # Lancement
    server.run()
