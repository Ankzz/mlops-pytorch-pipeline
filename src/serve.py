import io
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from model import get_model
from contextlib import asynccontextmanager
from dataset import get_serving_transform, CLASSES

# Global variables for model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = get_serving_transform()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    print("Loading ML model from /app/checkpoints...")
    # Load your model from the Kubernetes volume mount here
    # model = load_checkpoint("/app/checkpoints")
    load_model() # Replace with actual model
    print("Model loaded successfully.")
    
    yield  # This yields control back to FastAPI so it can start accepting requests
    
    # --- SHUTDOWN LOGIC ---
    print("Shutting down... cleaning up resources.")
    # Free up memory/GPU resources when the Kubernetes pod terminates
    global model
    model = None

app = FastAPI(title="CIFAR-10 Image Classifier API", lifespan=lifespan)

# @app.on_event("startup")
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

@app.get("/health")
def health_check():
    """
    Kubernetes Liveness and Readiness probes will call this every 5-10 seconds.
    """
    # Optional: Add logic here to check if your model is actually loaded into memory
    # if model is None:
    #     return JSONResponse(status_code=503, content={"status": "not ready"})
        
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    # Run the server locally on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)