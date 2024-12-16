from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
from torchvision.transforms import functional as F
from model import get_cnn_model, get_mask_rcnn_model

app = FastAPI()

# Load Models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = get_cnn_model().to(DEVICE)
cnn_model.load_state_dict(torch.load("models/cnn_vehicle_damage.pth"))
cnn_model.eval()

mask_rcnn_model = get_mask_rcnn_model(num_classes=2).to(DEVICE)
mask_rcnn_model.load_state_dict(torch.load("models/mask_rcnn_vehicle_damage.pth"))
mask_rcnn_model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...), model_type: str = "cnn"):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)

    if model_type == "cnn":
        outputs = cnn_model(img_tensor)
        prediction = torch.argmax(outputs, 1).cpu().item()
        return {"model": "CNN", "prediction": "Damaged" if prediction == 1 else "Undamaged"}
    
    if model_type == "mask_rcnn":
        outputs = mask_rcnn_model([img_tensor])[0]
        scores = outputs['scores'].cpu().numpy()
        result = any(scores > 0.5)
        return {"model": "Mask R-CNN", "prediction": "Damaged" if result else "Undamaged"}
