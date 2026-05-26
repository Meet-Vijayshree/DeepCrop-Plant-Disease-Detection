🌱 DeepCrop: A Deep Learning Framework for Multi-Plant Disease Detection
DeepCrop is an AI-powered plant disease detection system built using Convolutional Neural Networks (CNNs). It identifies multiple crop diseases using images and helps farmers take early action to protect crop health.

📌 Features
🌿 Detects diseases across multiple plant species
🧠 Uses a custom CNN deep learning model
📊 90%+ accuracy (depending on training configuration)
🗂 Trained on PlantVillage dataset
🖥 Simple and clean UI for uploading leaf images
📁 Well-organized dataset and training pipeline
🔍 Real-time prediction support
📂 Project Structure
DeepCrop/
│── dataset/
│   ├── train/
│   ├── test/
│   └── validation/
│
│── model/
│   ├── deepcrop_model.h5
│   └── model_training.ipynb
│
│── src/
│   ├── model.py
│   ├── predict.py
│   └── utils.py
│
│── webapp/
│   ├── app.py
│   ├── static/
│   ├── templates/
│   └── uploads/
│
│── README.md
│── requirements.txt
└── LICENSE
🔧 Tech Stack
Python
TensorFlow / Keras
NumPy, Pandas, Matplotlib
Flask (for Web App)
PlantVillage Dataset
🧠 Model Architecture (CNN)
Your CNN includes:

Convolution layers
MaxPooling layers
Flatten layer
Dense layers
Softmax output layer
Input → Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense → Output
📈 Training
To train the model:

python train.py
or open Jupyter Notebook:

jupyter notebook model_training.ipynb
🚀 Run the Web App
python app.py
Then open browser:

http://localhost:5000
🖼 Prediction
Upload a leaf image → Model predicts:

Healthy / Unhealthy
Name of disease
Confidence score
📥 Dataset
This project uses the PlantVillage dataset containing 50,000+ labelled images.

Download link (official): https://data.mendeley.com/datasets/tywbtsjrjv/1

📃 How to Train the Model? (Short Explanation)
Load dataset
Preprocess images (resize, normalization)
Split into train/test
Build CNN architecture
Compile model with Adam optimizer
Train 20–30 epochs
Save model
🧪 Results
Training Accuracy: ~95% (varies by crop)
Test Accuracy: ~90%
Low loss and stable validation curve
💡 Future Improvements
Add more crop classes
Implement MobileNet for faster predictions
Deploy on mobile app
Add voice-based explanation system

