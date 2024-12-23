import os
from flask import Flask, request, jsonify, render_template
import joblib  # Use joblib instead of pickle
from PIL import Image
import numpy as np
import cv2
from flask_cors import CORS

app = Flask(__name__, template_folder='template')
CORS(app)

# Load the trained model
model_path = "random_forest_model.joblib"
model = joblib.load(model_path)  # Use joblib to load the model

# Function to extract color histograms
def extract_color_histogram(image, bins=32):
    # Resize the image to 100x100 (to match the model's training size)
    resized_img = np.array(image.resize((100, 100)))
    
    # Convert to HSV color space
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2HSV)  # Fix the color channel conversion
    
    # Extract histograms for Hue, Saturation, and Value channels
    hist_hue = cv2.calcHist([hsv_img], [0], None, [bins], [0, 256]).flatten()
    hist_saturation = cv2.calcHist([hsv_img], [1], None, [bins], [0, 256]).flatten()
    hist_value = cv2.calcHist([hsv_img], [2], None, [bins], [0, 256]).flatten()
    
    # Concatenate histograms
    hist = np.concatenate([hist_hue, hist_saturation, hist_value])
    
    # Normalize the histogram
    hist = hist / hist.sum()  # Normalize to make it a probability distribution
    
    return hist

# Function to preprocess the image for the model
def preprocess_image(image):
    # Extract color histogram features (96 features)
    histogram = extract_color_histogram(image)
    
    # Ensure the image is in the correct shape: (1, 96)
    return np.expand_dims(histogram, axis=0)  # Add batch dimension (shape: 1, 96)

# Landing page route
@app.route("/")
def home():
    return render_template("index.html")  # Render the index.html template from the templates folder

# Prediction route
@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == 'POST':
        if 'fruitImage' not in request.files:
            return jsonify({'error': 'No file part provided in the request'}), 400

        file = request.files['fruitImage']
        if file.filename == '':
            return jsonify({'error': 'No file selected for upload'}), 400

        try:
            # Open the image file
            image = Image.open(file)
            
            # Preprocess the image
            preprocessed_image = preprocess_image(image)
            
            # Make the prediction
            prediction = model.predict(preprocessed_image)[0]  # Get prediction label directly
            
            # Return the prediction as a JSON response
            return jsonify({'prediction': prediction})

        except Exception as e:
            # Return JSON error response for better debugging
            return jsonify({'error': f"An error occurred: {str(e)}"}), 500

    # If GET request, render the prediction form page
    return render_template("prediction.html")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
