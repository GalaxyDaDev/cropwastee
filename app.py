from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import io
import requests

app = Flask(__name__)
CORS(app)

# URL of the model file
model_url = 'https://github.com/GalaxyDaDev/cropwastee/releases/download/ai/husk_model.pth'
model_path = 'husk_model.pth'

# Function to download the model file
def download_model(url, path):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    with open(path, 'wb') as f:
        f.write(response.content)

# Initialize model
def initialize_model():
    model = models.resnet18(weights='DEFAULT')
    num_classes = 8
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        download_model(model_url, model_path)
        model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = initialize_model()

# Define the class names
class_names = ['Chickpea Husk', 'Corn Husk', 'Field Pea Husk', 'Grass Pea Husk', 'Lentil Husk', 'Rice Husk', 'Wheat Husk', 'Soybean Husk']

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    image = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
