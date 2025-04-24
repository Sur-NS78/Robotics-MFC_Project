import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from picamera2 import Picamera2
import cv2
import logging
from datetime import datetime
import os

# Set up logging
log_dir = "pest_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"pest_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Set up frame save directory
frame_dir = "pest_frames"
os.makedirs(frame_dir, exist_ok=True)

# Load the pre-trained model
model = load_model('/home/raspi5/Downloads/pest_classification2.h5')
model.summary()
classes = ['beetle', 'caterpillar', 'earwig', 'grasshopper', 'slug', 'snail', 'weevil']
CONF_THRESHOLD = 0.8  # Confidence threshold for displaying predictions

def preprocess_image(img):
    """Preprocess the image for model prediction"""
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, [224, 224])
    img = np.expand_dims(img, axis=0)
    return img

def classify_frame(frame):
    """Classify the frame, log results, save frame, and annotate if confidence > 0.8"""
    # Preprocess and predict
    processed_frame = preprocess_image(frame)
    predictions = model.predict(processed_frame, verbose=0)  # Silent prediction
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    
    # Log the prediction
    logging.info(f"Predicted: {predicted_class}, Confidence: {confidence:.2f}")
    
    # Save the frame fed into the model (before annotation)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    frame_path = os.path.join(frame_dir, f"frame_{timestamp}.jpg")
    cv2.imwrite(frame_path, frame)  # Save original BGR frame
    
    # Only annotate if confidence exceeds threshold
    if confidence > CONF_THRESHOLD:
        # Create the label text
        label_text = f"Class: {predicted_class} (Confidence: {confidence:.2f})"
        
        # Get text size for annotation
        font_scale = 0.6
        font_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Define annotation position with padding
        padding = 10
        x, y = padding, 30  # Top-left corner position
        w_bg = text_width + 2 * padding
        h_bg = text_height + 2 * padding
        x_bg, y_bg = x, y - text_height - padding  # Adjust y_bg to position background above text
        
        # Ensure background stays within frame bounds
        y_bg = max(0, y_bg)
        if y_bg + h_bg > frame.shape[0] or x_bg + w_bg > frame.shape[1]:
            h_bg = min(h_bg, frame.shape[0] - y_bg)
            w_bg = min(w_bg, frame.shape[1] - x_bg)
        
        # Create small black background array
        black = np.zeros((h_bg, w_bg, 3), dtype=frame.dtype)
        
        # Blend semi-transparent background directly into frame slice
        alpha = 0.56
        cv2.addWeighted(black, alpha, frame[y_bg:y_bg+h_bg, x_bg:x_bg+w_bg], 1 - alpha, 0, 
                        dst=frame[y_bg:y_bg+h_bg, x_bg:x_bg+w_bg])
        
        # Add text on top
        cv2.putText(frame, label_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                    (0, 255, 0), font_thickness)  # Green text
    
    return frame

# Initialize the Raspberry Pi Camera Module
picam2 = Picamera2()

# Configure the camera
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)

# Start the camera
picam2.start()

try:
    while True:
        # Capture frame as numpy array
        frame = picam2.capture_array()
        
        # Convert from RGB (picamera2 default) to BGR (OpenCV default)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Classify and annotate the frame
        classified_frame = classify_frame(frame)
        
        # Display the result
        cv2.imshow('PiCam Pest Classification', classified_frame)
        
        # Exit on 'Esc' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()
