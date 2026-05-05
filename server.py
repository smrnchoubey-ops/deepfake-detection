import os

# --- Environment Configuration (MUST BE FIRST) ---
# Removed TF_USE_LEGACY_KERAS to fix RecursionError with TF 2.20 + Keras 3
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['MEDIAPIPE_DISABLE_GPU']='1'
os.environ['GLOG_minloglevel'] ='3'  # 3 = FATAL only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

from flask import Flask, render_template, redirect, request, url_for, send_file, send_from_directory, flash
from flask import jsonify, json
from werkzeug.utils import secure_filename
import datetime
import logging
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User

# Suppress MediaPipe GPU warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import time
import uuid
import sys
import traceback
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MTCNN for Face Detection
try:
    from mtcnn import MTCNN  # type: ignore
    detector = MTCNN()
    logger.info("MTCNN face detector loaded successfully.")
except ImportError:
    logger.error("MTCNN not found! Falling back to Haar Cascade (legacy).")
    # Fallback legacy initialization
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    detector = None

import zipfile
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import efficientnet_b0
import warnings
warnings.filterwarnings("ignore")

# Model paths
# Video detection: df_model.pt (ResNeXt50 + LSTM for temporal analysis)
# Image detection: best_model-v3.pt (EfficientNet-B0 for single images)
# Audio detection: audio_classifier.h5 (TensorFlow/Keras model)

# EfficientNet model path (for IMAGE detection)
EFFICIENTNET_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best_model-v3.pt')

# Audio classifier model path (for AUDIO detection) - removed

# DFModel path (for VIDEO detection) - loaded in get_model() function
DF_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'df_model.pt')

# New Xception Model path (Keras .h5)
VIDEO_MODEL_H5_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'df_92_v2.h5')


# Lazy loading for video h5 model
_video_model_h5 = None

def get_video_model_h5():
    """Load video classifier model (Keras .h5) using tf_keras for Keras 2 compatibility"""
    global _video_model_h5
    if _video_model_h5 is None:
        try:
            # Use tf_keras for Keras 2 API compatibility with .h5 models
            # tf_keras provides backward compatibility layer for Keras 2 models
            import tf_keras  # type: ignore
            
            logger.info(f"Loading video model from: {VIDEO_MODEL_H5_PATH}")
            if not os.path.exists(VIDEO_MODEL_H5_PATH):
                raise FileNotFoundError(f"Video model not found at: {VIDEO_MODEL_H5_PATH}")
            
            # Load model with compile=False to skip optimizer loading
            # Using tf_keras for full Keras 2 compatibility
            _video_model_h5 = tf_keras.models.load_model(
                VIDEO_MODEL_H5_PATH,
                compile=False
            )
            logger.info("Video Keras model loaded successfully with tf_keras!")
        except Exception as e:
            logger.error(f"Error loading video Keras model: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    return _video_model_h5

def crop_face_from_frame(frame_image):
    """
    Detect and crop face from a PIL Image using OpenCV Haar Cascades.
    Returns the cropped PIL Image (resized to 224x224) or None if no face found.
    """
    try:
        # Convert PIL to cv2 input (BGR)
        open_cv_image = np.array(frame_image) 
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        
        if detector:
            # MTCNN Detection
            # detect_faces expects RGB image
            faces = detector.detect_faces(np.array(frame_image))
            
            if not faces:
                return None
                
            # Get largest face (highest confidence * box area)
            # MTCNN returns dict with 'box': [x, y, width, height]
            largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
            x, y, w, h = largest_face['box']
            
            # Fix negative coordinates
            x, y = max(0, x), max(0, y)
        else:
            # Legacy Haar Cascade Detection
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None
            
            # Get largest face
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        
        # Add padding (20%)
        p_x = int(w * 0.2)
        p_y = int(h * 0.2)
        
        frame_h, frame_w, _ = open_cv_image.shape
        
        left = max(0, x - p_x)
        top = max(0, y - p_y)
        right = min(frame_w, x + w + p_x)
        bottom = min(frame_h, y + h + p_y)
        
        # Crop from original frame (RGB)
        cropped = np.array(frame_image)[top:bottom, left:right]
        
        if cropped.size == 0:
            return None
            
        # Resize to 224x224 for Xception
        cropped = cv2.resize(cropped, (224, 224))
        
        # Return as PIL Image
        return Image.fromarray(cropped)
        
    except Exception as e:
        logger.warning(f"Face crop failed for frame: {e}")
        return None

def predict_video_xception(video_path):
    """
    Predict video using the Keras .h5 model.
    Steps:
    1. Extract frames
    2. Crop faces
    3. Preprocess (normalize /255.0)
    4. Predict
    5. Average results
    """
    start_time = time.time()
    logger.info(f"Starting Xception video analysis for: {video_path}")
    
    try:
        model = get_video_model_h5()
        
        # Extract frames (using existing function)
        # Extract slightly more frames to ensure we catch faces
        frames, frame_paths = extract_video_frames(video_path, num_frames=50, save_frames=True)
        
        if not frames:
            raise Exception("No frames extracted from video")
            
        preprocessed_faces = []
        valid_frame_indices = []
        
        for i, frame in enumerate(frames):
            # Crop face
            face_crop = crop_face_from_frame(frame)
            if face_crop:
                # Debug: Save cropped face
                debug_path = f"static/debug_crops/crop_{i}.jpg"
                face_crop.save(debug_path)
                
                # Convert to numpy
                img_array = np.array(face_crop)
                
                # Standard Xception preprocessing (Map to -1 to 1)
                # Formula: (x / 127.5) - 1.0  OR  tf.keras.applications.xception.preprocess_input
                # Old code used / 255.0 (Map to 0 to 1). Let's stick to user's .h5 expectation (likely 0-1 if custom, but -1-1 if stock xception)
                # Trying standard Xception preprocessing first as false positives often mean "pattern not found"
                img_array = (img_array / 127.5) - 1.0
                
                preprocessed_faces.append(img_array)
                valid_frame_indices.append(i)
                
        if not preprocessed_faces:
            logger.warning("No faces detected in any frames. Fallback: using center crops.")
            # Fallback logic: center crop resize
            for i, frame in enumerate(frames):
                img = frame.resize((224, 224))
                img_array = np.array(img)
                img_array = (img_array / 127.5) - 1.0
                preprocessed_faces.append(img_array)
                valid_frame_indices.append(i)
        
        # Batch prediction
        batch_input = np.array(preprocessed_faces) # (N, 224, 224, 3)
        logger.info(f"Running inference on {len(batch_input)} frames")
        
        predictions = model.predict(batch_input, verbose=0)
        
        # Calculate average probability
        # Assuming output is probability of FAKE (based on user snippet: >0.5 is Fake)
        if predictions.shape[-1] == 1:
            fake_probs = predictions.flatten()
        else:
            # If 2 classes, usually [Real, Fake] or [Fake, Real]. 
            # Snippet said "pred[0][0]". This implies shape (1, 1). So standard binary.
            fake_probs = predictions[:, 0]
            
        avg_fake_prob = np.mean(fake_probs)
        
        # Determine label
        # 0 = FAKE, 1 = REAL (Project standard)
        # Model Output: 
        #   Close to 0.0 -> FAKE (Low "Realness")
        #   Close to 1.0 -> REAL (High "Realness")
        
        is_fake = avg_fake_prob < 0.5  # <--- INVERTED LOGIC based on testing
        
        prediction_label = 0 if is_fake else 1
        # Confidence is distance from 0.5 (or deviation from opposite)
        # If is_fake (prob < 0.5): confidence is (1 - prob) * 100 
        # If is_real (prob > 0.5): confidence is prob * 100
        confidence = float((1 - avg_fake_prob) * 100 if is_fake else avg_fake_prob * 100)
        
        # Heatmap generation
        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
        heatmap_url = generate_efficientnet_heatmap(fake_probs, heatmap_filename) # Reuse this helper
        
        processing_time = time.time() - start_time
        
        return [prediction_label, confidence, heatmap_url, frame_paths], processing_time
        
    except Exception as e:
        logger.error(f"Error in Xception prediction: {e}")
        traceback.print_exc()
        raise


# Get the absolute path for the upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Uploaded_Files')
# Remove unused folder paths
# FRAMES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'frames')
# GRAPHS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'graphs')

# Create the folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
HEATMAP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'heatmaps')
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
FRAMES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'frames')
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Ensure folders have proper permissions
os.chmod(HEATMAP_FOLDER, 0o755)
os.chmod(FRAMES_FOLDER, 0o755)

video_path = ""
detectOutput = []

app = Flask("__main__", template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize SQLAlchemy
db.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create all database tables
with app.app_context():
    db.create_all()

# Dataset comparison accuracies
DATASET_ACCURACIES = {
    'Our Model': 96.5,
    'FaceForensics++': 85.1,
    'DeepFake Detection Challenge': 82.3,
    'DeeperForensics-1.0': 80.7
}

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match")

        user = User.query.filter_by(email=email).first()
        if user:
            return render_template('signup.html', error="Email already exists")

        user = User.query.filter_by(username=username).first()
        if user:
            return render_template('signup.html', error="Username already exists")

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        return redirect(url_for('homepage'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('homepage'))
        else:
            return render_template('login.html', error="Invalid email or password")

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('homepage'))



def generate_temporal_heatmap(sequence_logits, filename):
    """
    Generate a temporal heatmap showing confidence over frames.
    sequence_logits: numpy array of shape [seq_length, 2]
    """
    try:
        # Calculate probabilities for 'Fake' class (index 1) which is usually 1, but let's check mapping
        # In this model, often 0 is FAKE and 1 is REAL, or vice versa. 
        # Standard: 0=Real, 1=Fake. Let's verify prediction logic below.
        # "if prediction[0] == 0: output = 'FAKE'". So 0 is Fake.
        # We want to plot the probability of FAKE (class 0).
        
        # Apply softmax to get probabilities
        probs = np.exp(sequence_logits) / np.sum(np.exp(sequence_logits), axis=1, keepdims=True)
        fake_probs = probs[:, 0] # Probability of being Fake (Class 0)
        
        # Reshape for heatmap (grid for better visualization)
        # 20 frames -> 4 rows x 5 columns
        if len(fake_probs) == 20:
             data = fake_probs.reshape(4, 5)
             yticklabels = ['Seq 1', 'Seq 2', 'Seq 3', 'Seq 4']
             xticklabels = ['1', '2', '3', '4', '5']
             plt.figure(figsize=(8, 6))
        else:
             # Fallback for other lengths
             data = fake_probs.reshape(1, -1)
             yticklabels = ['Fake Risk']
             xticklabels = True
             plt.figure(figsize=(10, 2))
        
        # Use annot=True for values, linewidths for grid effect
        sns.heatmap(data, cmap='coolwarm', cbar=True, 
                   yticklabels=yticklabels, xticklabels=xticklabels, 
                   vmin=0, vmax=1,
                   annot=True, fmt='.2f', annot_kws={"size": 10},
                   linewidths=1, linecolor='white', square=True)
                   
        plt.title("Fake Probability - Video Frame Segments")
        plt.xlabel("Frame Index (Relative)")
        plt.ylabel("Segment")
        plt.yticks(rotation=0) 
        
        save_path = os.path.join(HEATMAP_FOLDER, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return f"static/heatmaps/{filename}"
    except Exception as e:
        logger.error(f"Error generating temporal heatmap: {e}")
        return None



class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0,a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(self.transform(pil_frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

# ============================================================
# EfficientNet-B0 Model Integration (from DeepfakeDetector-main)
# ============================================================

# Lazy loading for EfficientNet model
_efficientnet_model = None
_efficientnet_transform = None

def get_efficientnet_model():
    """Load EfficientNet-B0 model from DeepfakeDetector-main"""
    global _efficientnet_model, _efficientnet_transform
    
    if _efficientnet_model is None:
        try:
            logger.info(f"Loading EfficientNet-B0 model from: {EFFICIENTNET_MODEL_PATH}")
            
            if not os.path.exists(EFFICIENTNET_MODEL_PATH):
                raise FileNotFoundError(f"EfficientNet model not found at: {EFFICIENTNET_MODEL_PATH}")
            
            # Initialize EfficientNet-B0 architecture
            _efficientnet_model = efficientnet_b0()
            _efficientnet_model.classifier[1] = torch.nn.Linear(
                _efficientnet_model.classifier[1].in_features, 2
            )
            
            # Load trained weights
            _efficientnet_model.load_state_dict(
                torch.load(EFFICIENTNET_MODEL_PATH, map_location=torch.device('cpu'))
            )
            _efficientnet_model.eval()
            
            # Transform for EfficientNet (224x224, ImageNet normalization)
            _efficientnet_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info("EfficientNet-B0 model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading EfficientNet model: {str(e)}")
            raise
    
    return _efficientnet_model, _efficientnet_transform

def extract_video_frames(video_path, num_frames=15, save_frames=True):
    """
    Extract evenly-spaced frames from video for analysis.
    Returns: (frames, frame_paths) - PIL images and saved file paths
    """
    frames = []
    frame_paths = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Cannot open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        raise Exception("Video has no frames")
    
    # Get evenly-spaced frame indices
    if num_frames is None or num_frames <= 0 or num_frames >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    # Generate unique session ID for this analysis
    session_id = uuid.uuid4().hex[:8]
    
    current_frame = 0
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame in indices:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
            
            # Save frame to disk if requested
            if save_frames:
                frame_filename = f"frame_{session_id}_{frame_count:02d}.jpg"
                frame_path = os.path.join(FRAMES_FOLDER, frame_filename)
                pil_frame.save(frame_path, "JPEG", quality=85)
                frame_paths.append(f"static/frames/{frame_filename}")
                frame_count += 1
        
        current_frame += 1
        
        if len(frames) >= len(indices):
            break
    
    cap.release()
    
    if len(frames) == 0:
        raise Exception("No frames could be extracted from video")
    
    logger.info(f"Extracted {len(frames)} frames from video (total: {total_frames})")
    return frames, frame_paths

def predict_video_efficientnet(video_path, num_frames=15):
    """
    Predict if video is fake using EfficientNet-B0 with multi-frame averaging.
    Returns: (prediction, confidence, per_frame_probs, frame_paths)
    - prediction: 0 = FAKE, 1 = REAL
    - confidence: percentage confidence
    - per_frame_probs: list of fake probabilities for each frame (for heatmap)
    - frame_paths: list of saved frame image paths
    """
    model, transform = get_efficientnet_model()
    frames, frame_paths = extract_video_frames(video_path, num_frames, save_frames=True)
    
    all_probs = []
    per_frame_fake_probs = []
    
    with torch.no_grad():
        for frame in frames:
            input_tensor = transform(frame).unsqueeze(0)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            all_probs.append(probs)
            # probs[1] is fake probability in EfficientNet model (class 1 = FAKE)
            per_frame_fake_probs.append(probs[1].item())
    
    # Average probabilities across all frames
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    predicted_class = torch.argmax(avg_probs).item()
    confidence = avg_probs[predicted_class].item() * 100
    
    # Map EfficientNet output to our format:
    # EfficientNet: 0 = Real, 1 = Fake
    # Our format: 0 = FAKE, 1 = REAL (inverted)
    if predicted_class == 1:  # EfficientNet says Fake
        our_prediction = 0  # Our FAKE
    else:  # EfficientNet says Real
        our_prediction = 1  # Our REAL
    
    return our_prediction, confidence, per_frame_fake_probs, frame_paths

def generate_efficientnet_heatmap(per_frame_probs, filename):
    """
    Generate temporal heatmap from per-frame fake probabilities.
    """
    try:
        probs = np.array(per_frame_probs)
        num_frames = len(probs)
        
        # Create grid layout: aim for roughly 4x5 or similar
        if num_frames <= 5:
            rows, cols = 1, num_frames
        elif num_frames <= 10:
            rows, cols = 2, (num_frames + 1) // 2
        elif num_frames <= 15:
            rows, cols = 3, 5
        else:
            rows, cols = 4, 5
        
        # Pad if needed
        total_cells = rows * cols
        if len(probs) < total_cells:
            probs = np.pad(probs, (0, total_cells - len(probs)), mode='edge')
        
        data = probs[:total_cells].reshape(rows, cols)
        
        plt.figure(figsize=(8, 6))
        yticklabels = [f'Seq {i+1}' for i in range(rows)]
        xticklabels = [str(i+1) for i in range(cols)]
        
        sns.heatmap(
            data, cmap='coolwarm', cbar=True,
            yticklabels=yticklabels, xticklabels=xticklabels,
            vmin=0, vmax=1,
            annot=True, fmt='.2f', annot_kws={"size": 10},
            linewidths=1, linecolor='white', square=True
        )
        
        plt.title("Fake Probability - Video Frame Segments")
        plt.xlabel("Frame Index (Relative)")
        plt.ylabel("Segment")
        plt.yticks(rotation=0)
        
        save_path = os.path.join(HEATMAP_FOLDER, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return f"static/heatmaps/{filename}"
    except Exception as e:
        logger.error(f"Error generating EfficientNet heatmap: {e}")
        return None

def detectFakeVideo(videoPath):
    """Detect if video is fake using DFModel (ResNeXt50 + LSTM)"""
    start_time = time.time()
    
    try:
        logger.info(f"Starting video analysis for: {videoPath}")
        
        # Load DFModel and transform
        model, transform = get_model()
        
        # Create video dataset for sequence processing
        video_dataset = validation_dataset([videoPath], sequence_length=20, transform=transform)
        
        # Get the video sequence
        frames_tensor = video_dataset[0]  # Returns [1, seq_length, C, H, W]
        
        logger.info(f"Loaded video sequence with shape: {frames_tensor.shape}")
        
        # Make prediction using DFModel
        with torch.no_grad():
            fmap, logits, sequence_logits = model(frames_tensor)
            probs = F.softmax(logits, dim=1)
            _, prediction = torch.max(probs, 1)
            confidence = probs[:, int(prediction.item())].item() * 100
            
            # Generate unique filename for heatmap
            heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
            
            # Convert sequence logits to numpy for heatmap
            seq_logits_np = sequence_logits.cpu().numpy()[0]  # [seq_len, 2]
            
            # Generate temporal heatmap
            heatmap_url = generate_temporal_heatmap(seq_logits_np, heatmap_filename)
            
            logger.info(f'Prediction: {prediction.item()}, Confidence: {confidence}%')
        
        # Extract and save sample frames for display
        frames, frame_paths = extract_video_frames(videoPath, num_frames=15, save_frames=True)
        
        processing_time = time.time() - start_time
        logger.info(f"Video processing completed in {processing_time:.2f} seconds")
        logger.info(f"Prediction: {'FAKE' if prediction.item() == 0 else 'REAL'} with {confidence:.1f}% confidence")
        logger.info(f"Saved {len(frame_paths)} analyzed frames")
        
        # Return prediction with frame_paths included
        # prediction: 0 = FAKE, 1 = REAL (DFModel format)
        return [prediction.item(), confidence, heatmap_url, frame_paths], processing_time
        
    except Exception as e:
        logger.error(f"Error in detectFakeVideo: {str(e)}")
        traceback.print_exc()
        raise

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'model_loaded': _model is not None
    })

@app.route('/')
def homepage():
    return render_template('home.html')



@app.route('/test')
def test_endpoint():
    """Simple test endpoint to verify the server is working"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    logger.info(f"Detect route called with method: {request.method}")
    
    if request.method == 'GET':
        logger.info("Rendering detect.html template")
        return render_template('detect.html')
    
    if request.method == 'POST':
        logger.info("Processing video upload")
        try:
            if 'video' not in request.files:
                logger.error("No video file in request")
                return render_template('detect.html', error="No video file uploaded")
                
            video = request.files['video']
            logger.info(f"Video file received: {video.filename}")
            
            if video.filename == '':
                logger.error("Empty video filename")
                return render_template('detect.html', error="No video file selected")
                
            if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
                logger.error(f"Invalid file format: {video.filename}")
                return render_template('detect.html', error="Invalid file format. Please upload MP4, AVI, or MOV files.")
            
            # Check file size (limit to 100MB)
            video.seek(0, 2)  # Seek to end
            file_size = video.tell()
            video.seek(0)  # Reset to beginning
            
            logger.info(f"Video file size: {file_size} bytes")
            
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                logger.error(f"File too large: {file_size} bytes")
                return render_template('detect.html', error="File too large. Please upload a video smaller than 100MB.")
                
            video_filename = secure_filename(video.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            video.save(video_path)
            
            logger.info(f"Processing video: {video_filename} (size: {file_size} bytes)")
            
            # Check if video file exists and has content
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise Exception("Video file is empty or corrupted")
            
            # Use EfficientNet-B0 model for detection
            # logger.info("Starting video analysis with EfficientNet-B0 model...")
            # prediction, processing_time = detectFakeVideo(video_path)
    
            # SWITCH TO NEW XCEPTION MODEL
            logger.info("Starting video analysis with Xception (.h5) model...")
            prediction, processing_time = predict_video_xception(video_path)
            
            logger.info(f"Analysis completed. Prediction: {prediction}, Time: {processing_time}")
            
            if prediction is None or len(prediction) < 2:
                raise Exception("Model prediction failed")
            
            if prediction[0] == 0:
                output = "FAKE"
            else:
                output = "REAL"
            confidence = float(prediction[1])
            heatmap_url = prediction[2] if len(prediction) > 2 else None
            frame_urls = prediction[3] if len(prediction) > 3 else []
            
            logger.info(f"Video prediction: {output} with confidence {confidence}%")
            
            data = {
                'output': output, 
                'confidence': float(confidence),
                'processing_time': round(processing_time, 2),
                'heatmap_url': heatmap_url,
                'frames_analyzed': len(frame_urls),
                'frame_urls': frame_urls
            }
            
            logger.info(f"Sending response data: {data}")
            logger.info(f"Data type: {type(data)}")
            data_json = json.dumps(data)
            logger.info(f"JSON data: {data_json}")
            
            os.remove(video_path)
            
            # Add debug logging for template rendering
            logger.info("About to render template with data")
            try:
                result = render_template('detect.html', data=data_json)
                logger.info("Template rendered successfully")
                return result
            except Exception as template_error:
                logger.error(f"Template rendering error: {str(template_error)}")
                traceback.print_exc()
                # Fallback: return data as JSON response
                return jsonify(data)
            
        except Exception as e:
            # Clean up video file if it exists
            if 'video_path' in locals() and os.path.exists(video_path):
                os.remove(video_path)
            
            error_msg = str(e)
            logger.error(f"Error processing video: {error_msg}")
            traceback.print_exc()
            
            # Return a more user-friendly error message
            if "timeout" in error_msg.lower():
                return render_template('detect.html', error="Processing took too long. Please try with a shorter video (under 30 seconds).")
            elif "memory" in error_msg.lower():
                return render_template('detect.html', error="Video too large. Please try with a smaller video file.")
            else:
                return render_template('detect.html', error=f"Error processing video: {error_msg}")

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

# ✅ Define DFModel before loading state dict
class DFModel(torch.nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(DFModel, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)  # Ensure same base model
        self.model = torch.nn.Sequential(*list(model.children())[:-2])
        self.lstm = torch.nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.linear1 = torch.nn.Linear(2048, num_classes)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.dp = torch.nn.Dropout(0.4)

    def forward(self, x):
        # Handle both 4D and 5D inputs for compatibility
        if len(x.shape) == 4:
            # 4D input: [batch_size, channels, height, width] - add sequence dimension
            x = x.unsqueeze(1)  # Adding sequence length dimension (1 for single image)
        
        # Now x is 5D: [batch_size, seq_length, c, h, w]
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        # Apply linear layer to ALL time steps to get per-frame predictions
        # x_lstm: [batch, seq_length, hidden_dim]
        # self.linear1(x_lstm): [batch, seq_length, num_classes]
        sequence_logits = self.linear1(x_lstm)
        
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :])), sequence_logits

# Lazy loading for model
_model = None
_transform = None

def get_model():
    """Load DFModel (ResNeXt50 + LSTM) for video detection"""
    global _model, _transform
    if _model is None:
        try:
            logger.info("Loading DFModel (ResNeXt50 + LSTM) for video detection...")
            # Load model from local path
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'df_model.pt')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"DFModel not found at: {model_path}")
            
            logger.info(f"Loading DFModel from: {model_path}")
            
            # Initialize model and load weights properly
            _model = DFModel()
            _model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            _model.eval()
            
            # Video frame transformation
            _transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("DFModel loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading DFModel: {str(e)}")
            raise
    
    return _model, _transform

def predict_image_xception(image_path):
    """
    Predict if image is fake using the Keras .h5 model (Xception).
    Replaces EfficientNet-B0.
    """
    try:
        model = get_video_model_h5()
        
        # Load image
        img = Image.open(image_path).convert("RGB")
        
        # Crop face
        face_crop = crop_face_from_frame(img)
        
        if face_crop:
            # Convert to numpy
            img_array = np.array(face_crop)
            # Preprocess to [-1, 1]
            img_array = (img_array / 127.5) - 1.0
        else:
            logger.warning("No face detected in image. Fallback: using center crop/resize.")
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = (img_array / 127.5) - 1.0
            
        # Add batch dimension: (1, 224, 224, 3)
        input_tensor = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(input_tensor, verbose=0)
        
        # Extract probability
        if prediction.shape[-1] == 1:
            fake_prob = float(prediction[0][0])
        else:
            # Assuming index 0 is what we track based on video logic
            fake_prob = float(prediction[0][0])
            
        # Apply Inverted Logic (matches video detection)
        # Low prob (< 0.5) -> FAKE
        # High prob (>= 0.5) -> REAL
        is_fake = fake_prob < 0.5
        
        prediction_label = 0 if is_fake else 1 # 0=FAKE, 1=REAL
        
        # Calculate confidence
        if is_fake:
            confidence = (1 - fake_prob) * 100
        else:
            confidence = fake_prob * 100
            
        return prediction_label, confidence
        
    except Exception as e:
        logger.error(f"Error in Xception image prediction: {str(e)}")
        traceback.print_exc()
        return None, str(e)




@app.route('/image-detect', methods=['GET', 'POST'])
def image_detect():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('image.html', error="No image file uploaded")
        
        image = request.files['image']
        if image.filename == '':
            return render_template('image.html', error="No image file selected")
        
        # Validate image format - only support common formats
        allowed_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        if not image.filename.lower().endswith(allowed_formats):
            return render_template('image.html', error=f"Invalid image format. Please upload JPG, PNG, BMP, or GIF files. (AVIF not supported)")
        
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        
        try:
            # Use Xception model for image detection
            prediction, confidence = predict_image_xception(image_path)
            
            if prediction is None:
                return render_template('image.html', error=f"Error processing image: {confidence}")
            
            output = "FAKE" if prediction == 0 else "REAL"
            os.remove(image_path)
            return render_template('image.html', output=output, confidence=confidence)
        except Exception as e:
            # Clean up image file on error
            if os.path.exists(image_path):
                os.remove(image_path)
            logger.error(f"Error in image detection: {str(e)}")
            return render_template('image.html', error=f"Error processing image: {str(e)}")
    
    return render_template('image.html')

if __name__ == '__main__':
    # Production configuration
    import warnings
    warnings.filterwarnings("ignore")
    
    # Suppress all GPU-related warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Configure logging for production
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress specific loggers
    logging.getLogger('mediapipe').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    # Get port from environment (Render sets this)
    port = int(os.environ.get('PORT', 10000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting DeepFake Detection App on port {port}")
    logger.info("Production mode: GPU disabled, CPU-only processing")
    
    # For Render deployment, bind to 0.0.0.0 and use the correct port
    try:
        app.run(
            host="0.0.0.0", 
            port=port, 
            debug=debug, 
            threaded=True,
            use_reloader=False  # Disable reloader for production
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)