from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import requests
import io

app = Flask(__name__)

# Load the model
model = models.resnet18(pretrained=False, num_classes=8)
model.load_state_dict(torch.hub.load_state_dict_from_url(
    'https://github.com/GalaxyDaDev/cropwastee/releases/download/ai/husk_model.pth',
    map_location='cpu'))
model.eval()

# Define transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image = Image.open(file.stream)
    image = preprocess(image)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        classes = ['Chickpea Husk', 'Corn Husk', 'Field Pea Husk', 'Grass Pea Husk', 'Lentil Husk', 'Rice Husk', 'Wheat Husk', 'Soybean Husk']
        predicted_class = classes[predicted.item()]
    
    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
