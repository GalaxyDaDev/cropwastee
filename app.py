from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
import os

app = Flask(__name__)

# URL to the model file
MODEL_URL = "https://github.com/GalaxyDaDev/cropwastee/releases/download/ai/husk_model.pth"
MODEL_PATH = "husk_model.pth"

# Download the model file if it does not exist
if not os.path.exists(MODEL_PATH):
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)

# Define the model architecture (assuming a simple feedforward network for example)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Example layers (replace with your model's architecture)
        self.fc1 = nn.Linear(3 * 224 * 224, 500)
        self.fc2 = nn.Linear(500, 7)

    def forward(self, x):
        x = x.view(-1, 3 * 224 * 224)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load the model
model = SimpleNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Define the classes
CLASSES = ['Chickpea Husk', 'Corn Husk', 'Field Pea Husk', 'Grass Pea Husk', 'Lentil Husk', 'Rice Husk', 'Wheat Husk']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        image = Image.open(file.stream)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            prediction = CLASSES[predicted.item()]
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
