from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload

# Load your trained model
MODEL_PATH = 'model/best_model_efficientnetb0.h5'
model = load_model(MODEL_PATH)

# Define class labels (change these based on your model)
class_labels = ['animal', 'cartoon', 'floral', 'floral fabric', 'geometry', 'ikat', 'plaid fabric', 'plain', 'polka', 'polka dot fabric', 'squares', 'striped fabric', 'stripes', 'tribal']  # Example labels

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')  # Optional: create this page

@app.route('/contact')
def contact():
    return render_template('contact.html')  # Optional: create this page

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get uploaded file
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions[0])]

            return render_template('predict.html', filename=filename, prediction=predicted_class)

    return render_template('predict.html', filename=None, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

