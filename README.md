# ✋ Hand Magic — Air Canvas

Draw in the air using your webcam and hand gestures!
Built with Python, MediaPipe, and OpenCV.

## Gestures
- ☝️ One finger = Draw
- ✌️ Two fingers = Hover (pause drawing)

## Setup
```bash
conda create -n handmagic python=3.10 -y
conda activate handmagic
pip install opencv-python mediapipe==0.10.9 numpy
python app.py
```

## Requirements
- Python 3.10
- Webcam