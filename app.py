from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from torchvision import models

app = Flask(__name__)

# Load the model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=8)
model.load_state_dict(torch.load('husk_model.pth', map_location='cpu'))
model.eval()

# Define transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class names
class_names = [
    'Chickpea Husk', 'Corn Husk', 'Field Pea Husk', 'Grass Pea Husk',
    'Lentil Husk', 'Rice Husk', 'Wheat Husk', 'Soybean Husk'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        img = Image.open(BytesIO(file.read())).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
