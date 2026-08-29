import argparse
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# CIFAR-10 class labels mapped to their index
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_trained_model(checkpoint_path: str):
    print(f"Loading model from {checkpoint_path}...")
    
    # 1. Initialize the base ResNet18 architecture
    model = models.resnet18(weights=None)
    
    # 2. Modify the final fully connected layer to output 10 classes (for CIFAR-10)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)
    
    # 3. Load your trained weights into the model
    # map_location='cpu' ensures it works even if you test on a machine without a GPU
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    # 4. Set the model to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()
    return model

def predict(model, image_path: str):
    # Standard preprocessing for CIFAR-10
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], 
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image)
        
        # PyTorch expects a batch dimension (B, C, H, W)
        # unsqueeze(0) changes shape from (3, 32, 32) to (1, 3, 32, 32)
        input_batch = input_tensor.unsqueeze(0) 

        # Run inference without tracking gradients (saves memory)
        with torch.no_grad():
            output = model(input_batch)
            
        # Get the index of the highest prediction score
        _, predicted_idx = torch.max(output, 1)
        
        class_name = CIFAR10_CLASSES[predicted_idx.item()]
        print(f"Prediction: {class_name}")
        return class_name

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained CIFAR-10 ResNet18 model")
    parser.add_argument("--checkpoint", type=str, default="models/resnet18_cifar10.pth", help="Path to the model .pth file")
    parser.add_argument("--image", type=str, required=True, help="Path to the test image")
    
    args = parser.parse_args()
    
    model = load_trained_model(args.checkpoint)
    predict(model, args.image)