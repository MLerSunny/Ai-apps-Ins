
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from src.model import DamageClassifier

app = FastAPI()

# Load Trained Model
MODEL_PATH = "models/damage_classifier.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DamageClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Transformations for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Predict function
def predict_damage(image):
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return "Damaged" if predicted.item() == 1 else "Undamaged"

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = predict_damage(image)
    return {"Prediction": result}
