let model;
let labels;
let webcamStream;
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const letterBox = document.getElementById('letter-box');
const detectedLetter = document.getElementById('detected-letter');

async function loadModel() {
    model = await tf.loadLayersModel('modelo/model.json');
    const labelsResponse = await fetch('modelo/labels.json');
    labels = await labelsResponse.json();
    console.log('✅ Modelo y labels cargados correctamente');
}

async function startCamera() {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = webcamStream;
    requestAnimationFrame(runDetection);
}

function stopCamera() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
    }
}

async function runDetection() {
    if (!model || !webcamStream) return;

    // Captura frame del video
    const tensor = tf.browser.fromPixels(video).resizeNearestNeighbor([64, 64]).mean(2).toFloat().expandDims(0).expandDims(-1);
    
    const prediction = model.predict(tensor);
    const index = prediction.argMax(-1).dataSync()[0];
    const letter = labels[index];
    detectedLetter.textContent = letter;

    // Limpiar canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Dibujar landmarks si los tienes (ejemplo ficticio)
    // Aquí puedes reemplazar con tus datos de landmarks si los obtienes de MediaPipe
    // Por ejemplo, dibujar puntos aleatorios para prueba:
    for (let i = 0; i < 21; i++) {
        const x = Math.random() * canvas.width;
        const y = Math.random() * canvas.height;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI); // puntos más finos
        ctx.fillStyle = 'red';
        ctx.fill();
    }

    requestAnimationFrame(runDetection);
}

// Botones
document.getElementById('startBtn').addEventListener('click', startCamera);
document.getElementById('stopBtn').addEventListener('click', stopCamera);

loadModel();
