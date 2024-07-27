from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import transforms, models
import requests
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

# Model setup
model_url = 'https://github.com/GalaxyDaDev/cropwastee/releases/download/ai/husk_model.pth'
model_path = 'husk_model.pth'

def download_model():
    if not os.path.isfile(model_path):
        print('Downloading model from GitHub...')
        response = requests.get(model_url)
        with open(model_path, 'wb') as file:
            file.write(response.content)
        print('Model downloaded successfully.')

download_model()
model = models.resnet18(weights='DEFAULT')
num_classes = 8
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_labels = ['Chickpea Husk', 'Corn Husk', 'Field Pea Husk', 'Grass Pea Husk', 'Lentil Husk', 'Rice Husk', 'Wheat Husk', 'Soybean Husk']

@app.route('/')
def index():
    return 'Flask API is running. Use /predict to make predictions.'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    image = Image.open(file.stream).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_labels[predicted.item()]
    
    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
