import os
import sys
import numpy as np
from PIL import Image
import torch # needed for direct import if not mocking

# Add current directory to path so we can import server
sys.path.append(os.getcwd())

# Define a dummy image creator if no image exists
def create_dummy_image(path):
    img = Image.new('RGB', (224, 224), color = 'red')
    img.save(path)
    return path

try:
    from server import predict_image_xception, crop_face_from_frame, get_video_model_h5
    
    # Create a dummy image for testing
    test_image_path = "test_image_verify.jpg"
    create_dummy_image(test_image_path)
    
    print(f"Testing image prediction on {test_image_path}...")
    
    # Run prediction
    prediction, confidence = predict_image_xception(test_image_path)
    
    print(f"Prediction: {prediction} (0=FAKE, 1=REAL)")
    print(f"Confidence: {confidence}")
    
    # Clean up
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
        
    print("Verification Successful!")

except Exception as e:
    print(f"Verification Failed: {e}")
    import traceback
    traceback.print_exc()
