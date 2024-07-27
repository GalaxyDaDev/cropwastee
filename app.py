from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18

app = Flask(__name__)

# Define the model architecture
class HuskModel(torch.nn.Module):
    def __init__(self):
        super(HuskModel, self).__init__()
        self.model = resnet18(pretrained=False)  # Example model
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 8)  # Adjust the final layer for 8 classes

    def forward(self, x):
        return self.model(x)

# Initialize and load the model
model = HuskModel()
model.load_state_dict(torch.hub.load_state_dict_from_url(
    'https://github.com/GalaxyDaDev/cropwastee/releases/download/ai/husk_model.pth', 
    map_location='cpu'))
model.eval()

# Define transformations for the input image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()

        # Replace with your class labels
        class_labels = ['Chickpea Husk', 'Corn Husk', 'Field Pea Husk', 
                         'Grass Pea Husk', 'Lentil Husk', 'Rice Husk', 'Wheat Husk']
        result = class_labels[class_idx]

        return jsonify({'class': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
