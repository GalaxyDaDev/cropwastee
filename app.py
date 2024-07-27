from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)  # To handle CORS issues in development

# Load the model
model = models.resnet18(pretrained=False)
num_classes = 8
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.hub.load_state_dict_from_url(
    'https://github.com/GalaxyDaDev/cropwastee/releases/download/ai/husk_model.pth',
    map_location=torch.device('cpu')
))
model.eval()

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class labels
class_names = [
    'Chickpea Husk', 'Corn Husk', 'Field Pea Husk', 'Grass Pea Husk',
    'Lentil Husk', 'Rice Husk', 'Wheat Husk', 'Soybean Husk'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': str(e)})

    image = preprocess(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()

    return jsonify({'class': class_names[class_idx]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
