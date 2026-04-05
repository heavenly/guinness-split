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
const CONF_THRESHOLD = 0.15; 
const IOU_THRESHOLD = 0.45;
const CLASS_NAMES = ['G', 'beer', 'glass'];

let model;
let isAnimating = false;

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

imageUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => processImage(img);
    img.src = URL.createObjectURL(file);
});

async function processImage(img) {
    if (!model || isAnimating) return;
    loadingOverlay.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    mainCanvas.width = img.width;
    mainCanvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    try {
        const { detections, padInfo } = await detectAndProcess(img);
        loadingOverlay.classList.add('hidden');
        animateResults(img, detections, padInfo);
    } catch (err) {
        console.error(err);
        loadingOverlay.classList.add('hidden');
    }
}

async function detectAndProcess(img) {
    return tf.tidy(() => {
        const image = tf.browser.fromPixels(img);
        const [h, w] = image.shape.slice(0, 2);
        const scale = Math.min(INPUT_SIZE / h, INPUT_SIZE / w);
        const newH = Math.round(h * scale);
        const newW = Math.round(w * scale);
        const padY = Math.floor((INPUT_SIZE - newH) / 2);
        const padX = Math.floor((INPUT_SIZE - newW) / 2);
        
        const resized = tf.image.resizeBilinear(image, [newH, newW]);
        const padded = resized.pad([[padY, INPUT_SIZE - newH - padY], [padX, INPUT_SIZE - newW - padX], [0, 0]]);
        const input = padded.div(255.0).transpose([2, 0, 1]).expandDims(0);
        
        const res = model.execute(input);
        const predictions = res.squeeze([0]).transpose([1, 0]);
        const boxes = predictions.slice([0, 0], [-1, 4]);
        const scores = predictions.slice([0, 4], [-1, 3]);
        
        const allDetections = [];
        const nmsBoxes = tf.tidy(() => {
            const [cx, cy, bw, bh] = tf.split(boxes, 4, 1);
            return tf.concat([cy.sub(bh.div(2)).div(640), cx.sub(bw.div(2)).div(640), cy.add(bh.div(2)).div(640), cx.add(bw.div(2)).div(640)], 1);
        });

        for (let c = 0; c < 3; c++) {
            const classScores = scores.slice([0, c], [-1, 1]).squeeze();
            const indices = tf.image.nonMaxSuppression(nmsBoxes, classScores, 5, IOU_THRESHOLD, CONF_THRESHOLD).arraySync();
            indices.forEach(idx => {
                allDetections.push({
                    box: boxes.slice([idx, 0], [1, 4]).arraySync()[0],
                    score: classScores.arraySync()[idx],
                    label: CLASS_NAMES[c]
                });
            });
        }
        return { detections: allDetections, padInfo: { padX, padY, scale } };
    });
}

function mapToCanvas(coord, isY, padInfo) {
    return isY ? ((coord - padInfo.padY) / padInfo.scale) : ((coord - padInfo.padX) / padInfo.scale);
}

function findExactMeniscus(img, gBox, beerBox, padInfo) {
    const tempCanvas = document.createElement('canvas');
    const tCtx = tempCanvas.getContext('2d');
    tempCanvas.width = img.width;
    tempCanvas.height = img.height;
    tCtx.drawImage(img, 0, 0);

    const [g_cx, g_cy, g_w, g_h] = gBox;
    const gLeft = mapToCanvas(g_cx - g_w/2, false, padInfo);
    const gWidth = g_w / padInfo.scale;
    const gHeight = g_h / padInfo.scale;

    let scanTop, scanBottom, yoloY = null;
    if (beerBox) {
        yoloY = mapToCanvas(beerBox[1] - beerBox[3]/2, true, padInfo);
        scanTop = Math.max(0, Math.round(yoloY - gHeight));
        scanBottom = Math.min(img.height - 1, Math.round(yoloY + gHeight));
    } else {
        const gTop = mapToCanvas(g_cy - g_h/2, true, padInfo);
        scanTop = Math.max(0, Math.round(gTop - gHeight));
        scanBottom = Math.min(img.height - 1, Math.round(gTop + gHeight * 2));
    }

    const scanX = Math.max(0, Math.round(gLeft + gWidth / 2));
    const scanWidth = Math.min(150, Math.round(gWidth * 1.5));
    const startX = Math.max(0, Math.min(img.width - scanWidth, scanX - scanWidth/2));
    
    const imageData = tCtx.getImageData(startX, scanTop, scanWidth, scanBottom - scanTop).data;
    let maxWeightedGradient = -1;
    let bestY = -1;

    for (let y = 1; y < (scanBottom - scanTop) - 1; y++) {
        let bPrev = 0, bNext = 0;
        for (let x = 0; x < scanWidth; x++) {
            const offP = ((y - 1) * scanWidth + x) * 4;
            const offN = ((y + 1) * scanWidth + x) * 4;
            bPrev += (imageData[offP] + imageData[offP+1] + imageData[offP+2]) / 3;
            bNext += (imageData[offN] + imageData[offN+1] + imageData[offN+2]) / 3;
        }
        
        const grad = (bPrev / scanWidth) - (bNext / scanWidth);
        const actualY = scanTop + y;
        
        // Weight gradient by proximity to YOLO prediction to avoid text interference
        const weight = yoloY !== null ? Math.exp(-Math.pow(actualY - yoloY, 2) / (2 * Math.pow(gHeight/2, 2))) : 1.0;
        const weightedGrad = grad * weight;

        if (weightedGrad > maxWeightedGradient) {
            maxWeightedGradient = weightedGrad;
            bestY = actualY;
        }
    }
    return bestY > 0 ? bestY : yoloY;
}

async function animateResults(img, detections, padInfo) {
    isAnimating = true;
    const gLogo = detections.find(d => d.label === 'G');
    const beerDet = detections.find(d => d.label === 'beer');

    let finalMeniscusY = null;
    if (gLogo) {
        finalMeniscusY = findExactMeniscus(img, gLogo.box, beerDet ? beerDet.box : null, padInfo);
    } else if (beerDet) {
        finalMeniscusY = mapToCanvas(beerDet.box[1] - beerDet.box[3]/2, true, padInfo);
    }

    const duration = 1200;
    let startTime = null;

    function frame(timestamp) {
        if (!startTime) startTime = timestamp;
        const progress = Math.min((timestamp - startTime) / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3);

        ctx.drawImage(img, 0, 0);

        if (finalMeniscusY !== null) {
            const currentY = finalMeniscusY * ease;
            ctx.strokeStyle = '#FFFFFF'; ctx.lineWidth = 4;
            ctx.beginPath(); ctx.moveTo(0, currentY); ctx.lineTo(mainCanvas.width, currentY); ctx.stroke();
            ctx.fillStyle = '#FFFFFF'; ctx.font = 'bold 20px Georgia'; ctx.fillText('MENISCUS', 20, currentY - 10);
        }

        if (progress < 1) {
            requestAnimationFrame(frame);
        } else {
            isAnimating = false;
            if (gLogo) {
                const [cx, cy, w, h] = gLogo.box;
                const targetY = mapToCanvas(cy, true, padInfo);
                const gX = mapToCanvas(cx - w/2, false, padInfo);
                const gW = w / padInfo.scale;

                ctx.setLineDash([5, 5]); ctx.strokeStyle = '#C0964D';
                ctx.beginPath(); ctx.moveTo(gX - 20, targetY); ctx.lineTo(gX + gW + 20, targetY); ctx.stroke();
                ctx.setLineDash([]); ctx.fillStyle = '#C0964D'; ctx.fillText('TARGET', gX + gW + 25, targetY + 5);

                if (finalMeniscusY !== null) {
                    calculateScore(targetY, finalMeniscusY, h / padInfo.scale);
                    gStatus.textContent = '✅ Logo Detected';
                    beerStatus.textContent = '✅ Precision Meniscus Found';
                }
            } else {
                scoreValue.textContent = 'N/A';
                gStatus.textContent = '❌ Logo Not Found';
            }
            resultsSection.classList.remove('hidden');
        }
    }
    requestAnimationFrame(frame);
}

function calculateScore(targetY, meniscusY, gHeight) {
    const diff = Math.abs(targetY - meniscusY);
    const norm_diff = diff / gHeight;
    let score = (meniscusY >= (targetY - gHeight/2) && meniscusY <= (targetY + gHeight/2)) 
        ? 3.75 + (1.25 * (1 - (diff / (gHeight/2))))
        : 3.75 * Math.exp(-2.0 * Math.pow(norm_diff, 2));

    scoreValue.textContent = score.toFixed(2);
    scoreValue.style.color = score > 4.5 ? '#ffd700' : (score > 3.0 ? '#EEE2D0' : '#B22222');
    scoreCommentary.textContent = score > 4.5 ? "ABSOLUTELY LEGENDARY!" : (score > 3.5 ? "Masterful work!" : "Practice makes perfect.");
}

cameraBtn.addEventListener('click', () => imageUpload.click());
init();
