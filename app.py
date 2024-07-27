import os
import torch
from flask import Flask, request, jsonify
from torchvision import models, transforms
from PIL import Image
import requests

app = Flask(__name__)

# Load the model
MODEL_URL = 'https://github.com/GalaxyDaDev/cropwastee/releases/download/ai/husk_model.pth'
MODEL_PATH = '/app/husk_model.pth'

# Download the model if not present
if not os.path.isfile(MODEL_PATH):
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)

model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()

# Define image transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class names
CLASS_NAMES = [
    'Chickpea Husk', 'Corn Husk', 'Field Pea Husk', 'Grass Pea Husk', 
    'Lentil Husk', 'Rice Husk', 'Wheat Husk'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load and preprocess the image
        image = Image.open(file.stream).convert('RGB')
        image = preprocess(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()

        # Return the prediction
        return jsonify({'class': CLASS_NAMES[class_idx]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
