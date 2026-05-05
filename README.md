# DeepFake Detection

A Flask web application for detecting deepfake videos and images using ResNet-50, LSTM, EfficientNet-B0.

## Features

- **Video Detection**: Upload videos (MP4, AVI, MOV) to detect deepfakes
- **Image Detection**: Analyze images for AI manipulation
- **Visual Analysis**: Temporal heatmaps showing frame-by-frame detection confidence
- **User Authentication**: Login/signup functionality

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python server.py
```

Then open `http://localhost:5000` in your browser.

## Project Structure

```
├── server.py           # Main Flask application
├── models.py           # Database models
├── models/             # AI model weights
│   └── best_model-v3.pt
├── templates/          # HTML templates
├── static/             # CSS and static files
└── requirements.txt    # Python dependencies
```

## Requirements

- Python 3.10 or 3.12
- See requirements.txt for all dependencies

## Dataset 

- Celeb V1 - https://www.kaggle.com/datasets/asifmzx/celeb-v1-df

- DF Challenge : https://www.kaggle.com/competitions/deepfake-detection-challenge

- FF++ : https://www.kaggle.com/datasets/hungle3401/faceforensics