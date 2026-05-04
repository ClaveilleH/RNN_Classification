const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { alpha: false });
const PIXEL_SIZE = 10;
let isDrawing = false;

// Configuration Chart.js
const chartCtx = document.getElementById('probChart').getContext('2d');
const probChart = new Chart(chartCtx, {
    type: 'bar',
    data: {
        labels: ['0','1','2','3','4','5','6','7','8','9'],
        datasets: [{ 
            data: new Array(10).fill(0), 
            backgroundColor: '#FACC15', // Couleur jaune pour les barres
            borderRadius: 4, 
            barThickness: 15 
        }]
    },
    options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { right: 40 } },
        scales: { 
            x: { display: false, max: 100 },
            y: { 
                grid: { display: false }, 
                border: { display: false }, 
                ticks: { color: '#9CA3AF', font: { weight: '600' } } 
            }
        },
        plugins: { legend: { display: false } }
    },
    plugins: [{
        id: 'labels',
        afterDatasetsDraw(chart) {
            const { ctx, data, chartArea: { left }, scales: { y } } = chart;
            ctx.save();
            ctx.font = 'bold 11px sans-serif';
            ctx.fillStyle = '#FACC15';
            data.datasets[0].data.forEach((val, i) => {
                const barWidth = chart.getDatasetMeta(0).data[i].width;
                ctx.fillText(val.toFixed(1) + '%', left + barWidth + 5, y.getPixelForTick(i) + 4);
            });
        }
    }]
});

// 
function init() {
    const overlay = document.getElementById('grid-overlay');
    for (let i = 0; i < 28 * 28; i++) {
        const div = document.createElement('div');
        div.className = 'cell';
        overlay.appendChild(div);
    }
    clearCanvas();
}

// transforme le mouvement brut de ta souris en une "peinture" précise sur le canevas
// e est un event de souris
function draw(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / PIXEL_SIZE);
    const y = Math.floor((e.clientY - rect.top) / PIXEL_SIZE);
    ctx.fillStyle = "white"; // Le dessin reste blanc (format MNIST classique)
    ctx.fillRect(x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
}

canvas.onmousedown = (e) => { isDrawing = true; draw(e); };
canvas.onmousemove = draw;
window.onmouseup = () => isDrawing = false;

function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    probChart.data.datasets[0].data = new Array(10).fill(0);
    probChart.update();
    document.getElementById('status').innerText = "Dessinez un chiffre";
}

async function predict() {
    const statusLabel = document.getElementById('status');
    const apiUrl = document.getElementById('apiUrl').value;
    
    try {
        statusLabel.innerText = "Analyse en cours...";
        
        const temp = document.createElement('canvas');
        temp.width = 28; temp.height = 28;
        temp.getContext('2d').drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);
        
        // 1. On crée l'objet payload => Un dictionnaire avec la cle image .
	// L'element est une image en format  png /Base64
	// récupérée à partir du canvas temp
        const payload = { image: temp.toDataURL() };
        
        // 2. On transforme en JSON => La chaine commence par data:image/png;base64, c'est le Data URI scheme.
	// Elle est divisée en quatre parties bien précises :
	// 0) data: : C'est le préfixe qui indique au navigateur :
	// "Attention, ceci n'est pas un lien vers un fichier distant,
	// les données arrivent directement dans le texte".
	// 1) image/png : C'est ce qu'on appelle le MIME type. Il indique au
	// navigateur (ou à ton serveur) le type de fichier qu'il doit reconstruire (ici, une image PNG).
	// 2) ;base64 : C'est la méthode de codage utilisée.
	// 3) , : La virgule sépare le "titre" de l'image (le type) du contenu réel de l'image
	// 4) iVBORw0KGgoAAA... : C'est la suite infinie de caractères qui représente l'image elle-même, encodée en Base64.
	const jsonPayload = JSON.stringify(payload);
        
        // 3. On affiche dans la console
        // console.log("Données envoyées :", jsonPayload);
	// Affiche juste la longueur pour vérifier que ce n'est pas vide
	console.log("Taille du payload :", jsonPayload.length + " caractères");
	// Affiche juste le début pour voir le format (le début du Base64)
	console.log("Début du JSON :", jsonPayload.substring(0, 100));
	// ou alors F12 puis l'onglet Réseau (Network) des outils de développement
	// pour voir la requete
	
        const res = await fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: jsonPayload // On utilise la variable ici
        });
	
	// Analyse du retour de la request fetch POST
        if (!res.ok) throw new Error("Erreur serveur");
	
        const data = await res.json();
        statusLabel.innerText = `Résultat : Chiffre ${data.digit}`;
        probChart.data.datasets[0].data = data.probabilities.map(p => p * 100);
        probChart.update();
	
    } catch (error) {
        statusLabel.innerText = "Erreur de connexion";
        console.error(error);
    }
}

init();
