from flask import Flask, render_template, request, jsonify
import re
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
import tempfile
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import psutil

# ─────────────────────────────────────────────
# Connexion PostgreSQL
# Dans Dokploy : définir la variable d'env DATABASE_URL
# Format : postgresql://user:password@host:5432/dbname
# ─────────────────────────────────────────────
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/mnist')

def get_db():
    """Retourne une connexion PostgreSQL."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def init_db():
    """Crée la table models si elle n'existe pas."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id          SERIAL PRIMARY KEY,
                    name        TEXT NOT NULL,
                    data        BYTEA NOT NULL,
                    created_at  TIMESTAMP DEFAULT NOW(),
                    is_active   BOOLEAN DEFAULT FALSE
                );
            """)
        conn.commit()
    print("[DB] Table 'models' prête.")


# ─────────────────────────────────────────────
class DigitClassifier:
    """Gère uniquement le modèle et l'inférence."""

    def __init__(self):
        self.model = None
        self.active_model_id = None  # id du modèle actuellement en mémoire
        self._sync_model()

    def _sync_model(self):
        """
        Vérifie l'id du modèle actif en BDD.
        Recharge en mémoire seulement si différent de ce qu'on a déjà.
        Appelé au démarrage ET avant chaque prédiction.
        """
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, name, data FROM models WHERE is_active = TRUE LIMIT 1")
                    row = cur.fetchone()

            if not row:
                print("[DB] Aucun modèle actif en base. En attente d'un upload.")
                self.model = None
                self.active_model_id = None
                return

            if row['id'] == self.active_model_id:
                return  # déjà le bon modèle en mémoire, rien à faire

            print(f"[DB] Nouveau modèle actif détecté : '{row['name']}' (id={row['id']})")
            self._load_model_from_bytes(bytes(row['data']))
            self.active_model_id = row['id']

        except Exception as e:
            print(f"[DB] Erreur sync modèle : {e}")

    def _load_model_from_bytes(self, model_bytes: bytes):
        """Écrit les bytes dans un fichier temporaire et charge le modèle Keras."""
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
            tmp.write(model_bytes)
            tmp_path = tmp.name
        try:
            self.model = load_model(tmp_path)
            print("[Model] Modèle chargé en mémoire.")
        finally:
            os.unlink(tmp_path)

    def load_model_from_bytes(self, model_bytes: bytes):
        """API publique pour recharger depuis des bytes (utilisée à l'upload)."""
        self._load_model_from_bytes(model_bytes)

    def _preprocess(self, base64_str):
        image_data = re.sub('^data:image/.+;base64,', '', base64_str)
        img = Image.open(BytesIO(base64.b64decode(image_data)))
        img = img.convert('L').resize((28, 28))
        return np.array(img).reshape(1, 28, 28, 1) / 255.0

    def predict(self, base64_image):
        self._sync_model()  # chaque replica vérifie si le modèle actif a changé
        if self.model is None:
            raise RuntimeError("Aucun modèle chargé. Uploadez-en un d'abord.")
        img_array = self._preprocess(base64_image)
        prediction = self.model.predict(img_array)[0]
        return {
            'digit':         int(np.argmax(prediction)),
            'confidence':    float(np.max(prediction)),
            'probabilities': prediction.tolist()
        }


# ─────────────────────────────────────────────
class DigitServer:
    """Gère le serveur web et les routes."""

    def __init__(self, classifier: DigitClassifier):
        self.app = Flask(__name__)
        self.classifier = classifier
        self._register_routes()

    def _register_routes(self):
        self.app.add_url_rule('/',                      'index',          self.index)
        self.app.add_url_rule('/predict',               'predict',        self.predict,          methods=['POST'])
        self.app.add_url_rule('/upload',                'upload_model',   self.upload_model,     methods=['POST'])
        self.app.add_url_rule('/models',                'list_models',    self.list_models,      methods=['GET'])
        self.app.add_url_rule('/models/<int:model_id>/activate',
                                                        'activate_model', self.activate_model,   methods=['POST'])
        self.app.add_url_rule('/models/<int:model_id>', 'delete_model',   self.delete_model,     methods=['DELETE'])
        self.app.add_url_rule('/metrics', 'metrics', self.metrics, methods=['GET'])
    # ── Routes ────────────────────────────────

    def index(self):
        return render_template('index.html')

    def predict(self):
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        try:
            result = self.classifier.predict(data['image'])
            return jsonify(result)
        except RuntimeError as e:
            return jsonify({'error': str(e)}), 503

    def upload_model(self):
        """
        Reçoit un fichier .keras, le stocke en BDD et l'active immédiatement.
        Paramètre form optionnel : name (ex: "cnn_v2")
        """
        if 'model' not in request.files:
            return jsonify({'error': 'Aucun fichier nommé "model" trouvé'}), 400

        file = request.files['model']
        if file.filename == '' or not file.filename.endswith('.keras'):
            return jsonify({'error': 'Format invalide (attendu .keras)'}), 400

        model_name = request.form.get('name', file.filename)
        model_bytes = file.read()

        # 1. Vérifier que le fichier est un modèle Keras valide
        try:
            self.classifier.load_model_from_bytes(model_bytes)
        except Exception as e:
            return jsonify({'error': f'Modèle invalide : {e}'}), 400

        # 2. Sauvegarder en BDD et activer
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    # Désactiver l'ancien modèle actif
                    cur.execute("UPDATE models SET is_active = FALSE WHERE is_active = TRUE")
                    # Insérer le nouveau modèle
                    cur.execute(
                        "INSERT INTO models (name, data, is_active) VALUES (%s, %s, TRUE) RETURNING id",
                        (model_name, psycopg2.Binary(model_bytes))
                    )
                    new_id = cur.fetchone()['id']
                conn.commit()

            return jsonify({
                'message':    f"Modèle '{model_name}' sauvegardé et activé.",
                'model_id':   new_id,
                'model_name': model_name
            })
        except Exception as e:
            return jsonify({'error': f'Erreur BDD : {e}'}), 500

    def list_models(self):
        """Retourne la liste de tous les modèles stockés (sans les données binaires)."""
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, name, created_at, is_active FROM models ORDER BY created_at DESC")
                    rows = cur.fetchall()
            # created_at n'est pas sérialisable JSON directement
            result = [
                {
                    'id':         r['id'],
                    'name':       r['name'],
                    'created_at': r['created_at'].isoformat(),
                    'is_active':  r['is_active']
                }
                for r in rows
            ]
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def activate_model(self, model_id: int):
        """Charge en mémoire un modèle existant en BDD et le marque comme actif."""
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT name, data FROM models WHERE id = %s", (model_id,))
                    row = cur.fetchone()

            if not row:
                return jsonify({'error': f'Modèle id={model_id} introuvable'}), 404

            # Charger en mémoire
            self.classifier.load_model_from_bytes(bytes(row['data']))

            # Mettre à jour is_active en BDD
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE models SET is_active = FALSE WHERE is_active = TRUE")
                    cur.execute("UPDATE models SET is_active = TRUE  WHERE id = %s", (model_id,))
                conn.commit()

            return jsonify({'message': f"Modèle '{row['name']}' (id={model_id}) activé."})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def delete_model(self, model_id: int):
        """Supprime un modèle de la BDD (pas possible s'il est actif)."""
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT is_active FROM models WHERE id = %s", (model_id,))
                    row = cur.fetchone()
                    if not row:
                        return jsonify({'error': 'Modèle introuvable'}), 404
                    if row['is_active']:
                        return jsonify({'error': 'Impossible de supprimer le modèle actif'}), 400
                    cur.execute("DELETE FROM models WHERE id = %s", (model_id,))
                conn.commit()
            return jsonify({'message': f'Modèle id={model_id} supprimé.'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def metrics(self):
        """
        Retourne les métriques système en temps réel.
        Appelé par le stress tester (ou tout outil de monitoring).
        """
        cpu_per_core = psutil.cpu_percent(percpu=True)
        mem          = psutil.virtual_memory()
        swap         = psutil.swap_memory()
 
        return jsonify({
            "cpu_percent":       psutil.cpu_percent(),          # CPU total (%)
            "cpu_per_core":      cpu_per_core,                  # liste par cœur
            "cpu_count":         psutil.cpu_count(),
            "mem_total_mb":      round(mem.total   / 1024**2),
            "mem_used_mb":       round(mem.used    / 1024**2),
            "mem_percent":       mem.percent,
            "swap_used_mb":      round(swap.used   / 1024**2),
            "swap_percent":      swap.percent,
            "load_avg_1m":       psutil.getloadavg()[0],        # charge 1 min
        })
 


    def run(self, debug=True):
        self.app.run(debug=debug, host='0.0.0.0')


# ─────────────────────────────────────────────
init_db()
classifier = DigitClassifier()
_server   = DigitServer(classifier)
app       = _server.app   # point d'entrée Gunicorn

if __name__ == '__main__':
    _server.run()