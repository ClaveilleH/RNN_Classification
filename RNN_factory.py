from tensorflow.keras import layers, models, datasets
import requests

def make_model(fn='model.keras') :
    # Charger MNIST
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0

    # Créer un modèle simple (CNN)
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=20)
    
    model.save('model.keras') # C'est ce fichier que Flask chargera

#====================================
def  upload_model(url = "http://127.0.0.1:5000/upload"):
    
    # Ouverture du fichier en mode binaire
    with open(fn, 'rb') as f:
        files = {'model': f}  # 'model' doit correspondre au nom dans request.files['model']
        
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            print("Succès :", response.json())
        else:
            print("Erreur :", response.text)

#====================================
if __name__ == '__main__':

    fn ='model.keras'
    #Genere le model 
    make_model(fn)

    # et le post au noeud
    #curl -X POST -F "file=@/chemin/vers/ton/nouveau_modele.keras" http://127.0.0.1:5000/upload
    #upload_model()
