# Guinness "Split the G" Scoring App

A web-based application to determine if you have successfully "Split the G" on your pint of Guinness.

## Features
- **Logo Detection**: Automatically identifies the Guinness 'G' on the glass.
- **Meniscus Detection**: Detects the level of the liquid (meniscus) relative to the logo.
- **Dynamic Scoring**: Provides a score from 0.00 to 5.00 based on the vertical alignment.
- **Visual Overlays**: Displays target lines and detected levels directly on the image.

## Tech Stack
- **TensorFlow.js**: For browser-side object detection.
- **YOLOv8n**: The underlying computer vision model.
- **Vanilla JS/HTML/CSS**: No heavy frameworks, fast and responsive.

## Getting Started
To run the application locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/heavenly/guinness-split.git
   ```
2. Serve the files using a local web server (required for model loading):
   ```bash
   # Using Python
   python3 -m http.server 8000
   # Or using Node.js
   npx http-server
   ```
3. Open `http://localhost:8000` in your browser.

## How it Works
1. **Pre-processing**: Images are resized to 640x640 and converted to NCHW format.
2. **Inference**: The YOLOv8 model detects the 'G' logo and the 'beer' level.
3. **Scoring**: The vertical distance between the center of the 'G' and the top of the beer level is calculated. An exponential decay function is applied to generate the final score.

## Model Details
The model was trained to detect:
- `G`: The Guinness logo.
- `beer`: The liquid stout level.
- `glass`: The pint glass outline.

Built with YOLOv8 & TensorFlow.js.
