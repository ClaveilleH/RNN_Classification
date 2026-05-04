FROM python:3.11-slim
 
WORKDIR /app
 
# Dépendances système pour Pillow et TensorFlow
RUN apt-get update && apt-get install -y     libgl1     libglib2.0-0     && rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY . .
 
# Dossier pour les modèles uploadés
RUN mkdir -p models
 
EXPOSE 5000
 
CMD [python, app.py]
 

