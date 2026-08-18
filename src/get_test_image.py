import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_cifar_batch(file_path):
    """Load a single CIFAR-10 batch file."""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        return data, labels

def main():
    try:
        # Path to your CIFAR-10 folder
        cifar_dir = "data/cifar-10-batches-py"

        # Load test batch
        test_data, test_labels = load_cifar_batch(os.path.join(cifar_dir, "test_batch"))

        # Pick an image index
        index = 0
        img_flat = test_data[index]
        label = test_labels[index]

        # Reshape and convert to RGB image
        img_reshaped = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)

        # Save image
        os.makedirs("cifar10_samples", exist_ok=True)
        file_path = f"cifar10_samples/test_image_{index}.png"
        plt.imsave(file_path, img_reshaped.astype(np.uint8))

        print(f"✅ Saved CIFAR-10 test image to: {file_path}")
        print(f"🖼 Class label: {CLASS_NAMES[label]} (class index: {label})")

    except FileNotFoundError:
        print("❌ CIFAR-10 data not found. Please check the 'cifar_dir' path.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()