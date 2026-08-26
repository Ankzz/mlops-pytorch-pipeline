import io
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from model import get_model
from dataset import get_serving_transform, CLASSES

app = FastAPI(title="CIFAR-10 Image Classifier API")

# Global variables for model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = get_serving_transform()

@app.on_event("startup")
def load_model():
    """Loads the model weights when the server starts."""
    global model
    model = get_model(num_classes=10)
    
    # Load weights (ensure train.py has been run first to generate this file)
    try:
        model.load_state_dict(torch.load("models/resnet18_cifar10.pth", map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("WARNING: resnet18_cifar10.pth not found. Run train.py first.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts an image file and returns the predicted class."""
    # Read the image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    class_name = CLASSES[predicted_idx.item()]
    confidence_score = confidence.item()
    
    return {
        "filename": file.filename,
        "prediction": class_name,
        "confidence": round(confidence_score * 100, 2)
    }

if __name__ == "__main__":
    import uvicorn
    # Run the server locally on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)