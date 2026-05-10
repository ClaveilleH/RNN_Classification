from tensorflow.keras import layers, models, datasets
import requests

# ─────────────────────────────────────────────
# URL de ton serveur Dokploy — change cette valeur
# ─────────────────────────────────────────────
SERVER_URL = "http://rnn.dokpoly.claveille.fr"


def make_model(fn='model.keras'):
    """Entraîne un CNN sur MNIST et sauvegarde le fichier .keras."""
    (x_train, y_train), _ = datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20)
    model.save(fn)
    print(f"[Factory] Modèle sauvegardé : {fn}")

def make_model_v2(fn='model_v2.keras'):
    (x_train, y_train), _ = datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),  # 2ème couche conv
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),                          # évite l'overfitting
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    model.save(fn)
    print(f"[Factory] Modèle sauvegardé : {fn}")

def upload_model(fn='model.keras', name=None, url=None):
    """
    Envoie le fichier .keras au serveur Dokploy.
    Le serveur le stocke en base de données et l'active immédiatement.

    Paramètres :
        fn   : chemin local du fichier .keras
        name : nom lisible pour identifier ce modèle en BDD (optionnel)
        url  : URL complète du endpoint /upload (par défaut SERVER_URL)
    """
    if url is None:
        url = f"{SERVER_URL}/upload"
    if name is None:
        name = fn  # utilise le nom de fichier par défaut

    with open(fn, 'rb') as f:
        files = {'model': f}
        data  = {'name': name}
        response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        print("[Upload] Succès :", response.json())
    else:
        print("[Upload] Erreur :", response.text)


def list_models(url=None):
    """Affiche tous les modèles stockés sur le serveur."""
    if url is None:
        url = f"{SERVER_URL}/models"
    response = requests.get(url)
    if response.status_code == 200:
        models_list = response.json()
        print(f"\n{'ID':<5} {'Actif':<7} {'Nom':<30} {'Créé le'}")
        print("-" * 65)
        for m in models_list:
            active = "✓" if m['is_active'] else " "
            print(f"{m['id']:<5} {active:<7} {m['name']:<30} {m['created_at']}")
    else:
        print("[Erreur]", response.text)


def activate_model(model_id: int, url=None):
    """Demande au serveur de charger et activer un modèle existant en BDD."""
    if url is None:
        url = f"{SERVER_URL}/models/{model_id}/activate"
    response = requests.post(url)
    if response.status_code == 200:
        print("[Activate] Succès :", response.json())
    else:
        print("[Activate] Erreur :", response.text)


# ─────────────────────────────────────────────
if __name__ == '__main__':
    # fn = 'model.keras'

    # 1. Entraîner le modèle localement
    # make_model(fn)

    # 2. L'uploader sur Dokploy (sauvegarde en BDD + activation)
    # upload_model(fn, name='cnn_v1')

    # 3. Lister les modèles disponibles sur le serveur
    # list_models()

    # 4. (Optionnel) Activer un ancien modèle par son ID
    # activate_model(1)

    # make_model_v2('model_v2.keras')
    # upload_model('model_v2.keras', name='cnn_v2_dropout')
    # list_models()
    activate_model(2)