# 🐾 PawScan AI — Dog Skin Disease Detection

![PawScan AI Banner](https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80)

**PawScan AI** is a deep learning-powered web application that helps pet owners and veterinarians identify common dog skin diseases from images. It uses a **MobileNetV2** model trained on a custom dataset to classify skin conditions with high accuracy and includes a **RAG-based Chatbot** for answering queries about symptoms, treatments, and prevention.

## 🚀 Features

- **📸 AI Disease Detection**: Upload an image of a dog's skin, and the model predicts the condition with confidence scores.
- **🔍 6 Detected Classes**:
  - **Dermatitis**
  - **Fungal Infections**
  - **Healthy Skin**
  - **Hypersensitivity / Allergic Dermatosis**
  - **Demodicosis (Demodectic Mange)**
  - **Ringworm (Dermatophytosis)**
- **🤖 Intelligent Chatbot**: A RAG (Retrieval-Augmented Generation) chatbot uses TF-IDF to retrieve accurate answers from a curated veterinary knowledge base.
- **📚 Comprehensive Knowledge Base**: Detailed pages for each disease including symptoms, causes, treatments, and prevention tips.
- **✨ Premium UI**: Modern, dark-themed interface with glassmorphism effects and responsive design.

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Deep Learning**: TensorFlow, Keras (MobileNetV2 Transfer Learning)
- **Natural Language Processing**: Scikit-Learn (TF-IDF Vectorizer for Chatbot)
- **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript (Vanilla)
- **Data Processing**: NumPy, Pillow

## 📂 Project Structure

```
Dog-Skin-Disease-Detection/
├── app.py                 # Main Flask application
├── chatbot.py             # RAG Chatbot engine (TF-IDF)
├── train_model.py         # Script used to train the MobileNetV2 model
├── best_model.h5          # Trained TensorFlow model file
├── class_names.json       # Class labels mapping
├── disease_data.json      # Knowledge base for diseases and chatbot
├── requirements.txt       # Python dependencies
├── static/                # CSS, JS, and uploaded images
│   ├── css/style.css
│   └── js/main.js
└── templates/             # HTML Templates
    ├── index.html
    ├── result.html
    ├── disease.html
    └── chat_widget.html
```

## ⚙️ Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/Pawann30/Dog-Skin-Inefection-Prediction.git
   cd Dog-Skin-Inefection-Prediction
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Access the App**
   Open your browser and navigate to `http://localhost:5000`.

## 🧠 Model Training

The model was trained using **Transfer Learning** with the **MobileNetV2** architecture (pre-trained on ImageNet). 
- **Dataset**: Custom dataset with 6 classes.
- **Preprocessing**: Images resized to 224x224, normalized to [0,1], and augmented (rotation, zoom, flip).
- **Performance**: Achieved **~89% accuracy** on the test set.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
