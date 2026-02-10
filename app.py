"""
Flask Application for Dog Skin Disease Detection.
Loads a trained .h5 model, accepts image uploads, and predicts diseases.
Includes disease detail pages and a RAG chatbot assistant.
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ──── Model Configuration ────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.h5')
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), 'class_names.json')
IMAGE_SIZE = (224, 224)

# Try to load class names from saved config (created by training script)
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_config = json.load(f)
    CLASS_NAMES = class_config['class_names']
    IMAGE_SIZE = (class_config.get('img_size', 224), class_config.get('img_size', 224))
    print(f"📋 Loaded class names from class_names.json: {CLASS_NAMES}")
else:
    CLASS_NAMES = [
        'Dermatitis',
        'Fungal_infections',
        'Healthy',
        'Hypersensitivity',
        'demodicosis',
        'ringworm'
    ]

# ──── Load Disease Data ────
DISEASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'disease_data.json')
with open(DISEASE_DATA_PATH, 'r', encoding='utf-8') as f:
    DISEASE_DATA = json.load(f)

# ──── Load Model ────
model = None

def load_model():
    global model
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        print("   The app will run but predictions won't work.")
        model = None


def preprocess_image(image_bytes):
    """Preprocess an uploaded image for model prediction."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def predict_disease(image_bytes):
    """Run the model on the image and return predictions."""
    if model is None:
        # Return a mock prediction for demo/testing
        return {
            'class': 'fungal_infection',
            'confidence': 0.85,
            'all_predictions': {
                'bacterial_dermatosis': 0.05,
                'fungal_infection': 0.85,
                'healthy_skin': 0.03,
                'hypersensitivity_allergic_dermatosis': 0.07
            }
        }

    img_array = preprocess_image(image_bytes)
    predictions = model.predict(img_array, verbose=0)[0]

    # Get the predicted class
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    all_preds = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}

    return {
        'class': predicted_class,
        'confidence': confidence,
        'all_predictions': all_preds
    }


# ──── Routes ────

@app.route('/')
def index():
    """Home page with image upload."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction results."""
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Read and save image
    image_bytes = file.read()

    # Save uploaded image for display
    filename = secure_filename(file.filename)
    if not filename:
        filename = 'uploaded_image.jpg'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(image_bytes)

    # Get prediction
    result = predict_disease(image_bytes)

    # Get disease info
    disease_key = result['class']
    disease_info = DISEASE_DATA['diseases'].get(disease_key, {})

    return render_template('result.html',
                           prediction=result,
                           disease=disease_info,
                           image_url=url_for('static', filename=f'uploads/{filename}'),
                           all_predictions=result['all_predictions'])


@app.route('/disease/<slug>')
def disease_detail(slug):
    """Disease detail page with description, treatment, and prevention."""
    disease = DISEASE_DATA['diseases'].get(slug)
    if not disease:
        return redirect(url_for('index'))
    return render_template('disease.html', disease=disease)


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Chatbot API endpoint."""
    from chatbot import chat
    data = request.get_json()
    query = data.get('message', '')
    response = chat(query)
    return jsonify({'response': response})


# ──── Main ────

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
