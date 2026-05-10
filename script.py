import subprocess
import requests
import time

# Configuration
DOKPLOY_URL = "https://ton-dokploy.com"
API_TOKEN = "ton-api-token"
APP_ID = "rnn-server-rhyjyj"

MIN_REPLICAS = 1
MAX_REPLICAS = 5
SCALE_UP_THRESHOLD = 80    # % CPU pour scaler up
SCALE_DOWN_THRESHOLD = 30  # % CPU pour scaler down
CHECK_INTERVAL = 30        # secondes entre chaque vérification

def get_cpu_usage():
    """Lit l'utilisation CPU actuelle du serveur"""
    result = subprocess.run(
        ["grep", "cpu ", "/proc/stat"],
        capture_output=True, text=True
    )
    # Calcule le % CPU sur 1 seconde
    fields = list(map(int, result.stdout.split()[1:]))
    idle = fields[3]
    total = sum(fields)
    time.sleep(1)
    result2 = subprocess.run(
        ["grep", "cpu ", "/proc/stat"],
        capture_output=True, text=True
    )
    fields2 = list(map(int, result2.stdout.split()[1:]))
    idle2 = fields2[3]
    total2 = sum(fields2)
    return 100 * (1 - (idle2 - idle) / (total2 - total))

def get_current_replicas():
    """Demande à Dokploy combien de réplicas tournent actuellement"""
    response = requests.get(
        f"{DOKPLOY_URL}/api/application.one",
        params={"applicationId": APP_ID},
        headers={"Authorization": f"Bearer {API_TOKEN}"}
    )
    return response.json()["replicas"]

def set_replicas(n):
    """Dit à Dokploy de passer à n réplicas et redéploie"""
    requests.post(
        f"{DOKPLOY_URL}/api/application.update",
        json={"applicationId": APP_ID, "replicas": n},
        headers={"Authorization": f"Bearer {API_TOKEN}"}
    )
    requests.post(
        f"{DOKPLOY_URL}/api/application.redeploy",
        json={"applicationId": APP_ID},
        headers={"Authorization": f"Bearer {API_TOKEN}"}
    )
    print(f"→ Passage à {n} réplicas")

def autoscaler():
    print("Autoscaler démarré")
    while True:
        cpu = get_cpu_usage()
        replicas = get_current_replicas()
        print(f"CPU: {cpu:.1f}% | Réplicas: {replicas}")

        if cpu > SCALE_UP_THRESHOLD and replicas < MAX_REPLICAS:
            set_replicas(replicas + 1)

        elif cpu < SCALE_DOWN_THRESHOLD and replicas > MIN_REPLICAS:
            set_replicas(replicas - 1)

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    autoscaler()

"""
Les limites de cette approche
Le CPU seul c'est un indicateur imparfait — dans ton cas avec TensorFlow, une inférence peut saturer le CPU brièvement
 sans que le service soit vraiment surchargé. En production on surveillerait plutôt le nombre de requêtes en attente ou
   le temps de réponse moyen, ce qui est plus représentatif de la charge réelle"""