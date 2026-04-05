// Elements
const imageUpload = document.getElementById('image-upload');
const cameraBtn = document.getElementById('camera-btn');
const mainCanvas = document.getElementById('main-canvas');
const ctx = mainCanvas.getContext('2d');
const modelStatus = document.getElementById('model-status');
const loadingOverlay = document.getElementById('loading-overlay');
const resultsSection = document.getElementById('results');
const scoreValue = document.getElementById('score-value');
const scoreCommentary = document.getElementById('score-commentary');
const gStatus = document.getElementById('g-status');
const beerStatus = document.getElementById('beer-status');

// Configuration
const MODEL_PATH = './model.json';
const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;
const CLASS_NAMES = ['G', 'beer', 'glass'];

let model;

// Load Model
async function init() {
    try {
        modelStatus.textContent = 'Loading Model...';
        model = await tf.loadGraphModel(MODEL_PATH);
        modelStatus.textContent = 'Model Ready';
        modelStatus.style.backgroundColor = '#2e7d32';
    } catch (err) {
        console.error('Failed to load model:', err);
        modelStatus.textContent = 'Error loading model';
        modelStatus.style.backgroundColor = '#c62828';
    }
}

// Handle Image Upload
imageUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => processImage(img);
    img.src = URL.createObjectURL(file);
});

// Process Image
async function processImage(img) {
    if (!model) {
        alert('Model not loaded yet!');
        return;
    }

    loadingOverlay.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    // Setup canvas
    mainCanvas.width = img.width;
    mainCanvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    // Run inference
    const detections = await detect(img);
    
    // Draw and score
    await renderResults(img, detections);
    
    loadingOverlay.classList.add('hidden');
    resultsSection.classList.remove('hidden');
}

// Inference logic
async function detect(img) {
    return tf.tidy(() => {
        const tensor = tf.browser.fromPixels(img)
            .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
            .toFloat()
            .div(255.0);

        const input = tensor.transpose([2, 0, 1]).expandDims(0);
        const res = model.execute(input); // [1, 7, 8400]
        const predictions = res.squeeze([0]).transpose([1, 0]);
        
        const boxes = predictions.slice([0, 0], [-1, 4]);
        const scores = predictions.slice([0, 4], [-1, 3]);
        
        const maxScores = scores.max(1);
        const classIndices = scores.argMax(1);
        
        return {
            boxes: boxes.arraySync(),
            scores: maxScores.arraySync(),
            classes: classIndices.arraySync()
        };
    });
}

// Non-Max Suppression and Scoring
async function renderResults(img, rawDetections) {
    const { boxes, scores, classes } = rawDetections;
    const filteredDetections = [];
    
    const boxesTensor = tf.tensor2d(boxes.map(b => [
        (b[1] - b[3]/2) / INPUT_SIZE, 
        (b[0] - b[2]/2) / INPUT_SIZE, 
        (b[1] + b[3]/2) / INPUT_SIZE, 
        (b[0] + b[2]/2) / INPUT_SIZE
    ]));
    const scoresTensor = tf.tensor1d(scores);
    
    const nmsIndices = await tf.image.nonMaxSuppressionAsync(
        boxesTensor, 
        scoresTensor, 
        20, 
        IOU_THRESHOLD, 
        CONF_THRESHOLD
    );
    
    const indices = nmsIndices.arraySync();
    indices.forEach(idx => {
        filteredDetections.push({
            box: boxes[idx],
            score: scores[idx],
            label: CLASS_NAMES[classes[idx]]
        });
    });

    const gLogo = filteredDetections.find(d => d.label === 'G');
    const beerLevel = filteredDetections.find(d => d.label === 'beer');

    // Visuals
    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = 4;
    ctx.font = 'bold 20px Georgia';
    
    if (gLogo) {
        const [cx, cy, w, h] = gLogo.box;
        const targetY = (cy / INPUT_SIZE) * mainCanvas.height;
        const gX = ((cx - w/2) / INPUT_SIZE) * mainCanvas.width;
        const gW = (w / INPUT_SIZE) * mainCanvas.width;

        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = '#C0964D';
        ctx.beginPath();
        ctx.moveTo(gX - 20, targetY);
        ctx.lineTo(gX + gW + 20, targetY);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#C0964D';
        ctx.fillText('TARGET', gX + gW + 25, targetY + 5);
    }

    if (beerLevel) {
        const [cx, cy, w, h] = beerLevel.box;
        const currentY = ((cy - h/2) / INPUT_SIZE) * mainCanvas.height;
        ctx.strokeStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.moveTo(0, currentY);
        ctx.lineTo(mainCanvas.width, currentY);
        ctx.stroke();
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText('MENISCUS', 20, currentY - 10);
    }

    if (gLogo && beerLevel) {
        calculateScore(gLogo, beerLevel);
        gStatus.textContent = '✅ Logo Detected';
        beerStatus.textContent = '✅ Beer Level Detected';
    } else {
        scoreValue.textContent = 'N/A';
        scoreCommentary.textContent = 'Could not find both the G and the beer level. Try a clearer photo!';
        gStatus.textContent = gLogo ? '✅ Logo Detected' : '❌ Logo Not Found';
        beerStatus.textContent = beerLevel ? '✅ Beer Level Detected' : '❌ Beer Level Not Found';
    }
}

function calculateScore(gLogo, beerLevel) {
    const [g_cx, g_cy, g_w, g_h] = gLogo.box;
    const [b_cx, b_cy, b_w, b_h] = beerLevel.box;

    const g_center_y = g_cy; 
    const meniscus_y = b_cy - b_h / 2;
    const diff = Math.abs(g_center_y - meniscus_y);
    const norm_diff = diff / g_h;
    
    let score = 5.0 * Math.exp(-6 * norm_diff);
    scoreValue.textContent = score.toFixed(2);
    
    if (score > 4.5) {
        scoreCommentary.textContent = "ABSOLUTELY LEGENDARY!";
        scoreValue.style.color = '#ffd700'; 
    } else if (score > 3.5) {
        scoreCommentary.textContent = "Excellent work!";
        scoreValue.style.color = '#EEE2D0';
    } else if (score > 2.0) {
        scoreCommentary.textContent = "A solid attempt.";
        scoreValue.style.color = '#C0964D';
    } else {
        scoreCommentary.textContent = "Practice makes perfect.";
        scoreValue.style.color = '#B22222';
    }
    return score;
}

cameraBtn.addEventListener('click', () => {
    imageUpload.click();
});

// Init
init();
