import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR-10 classes for reference
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_serving_transform():
    """Transform used for inference/serving."""
    return transforms.Compose([
        transforms.Resize((32, 32)), # Reducing size from 244x244 to 32x32 as execution is on CPU
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_dataloaders(batch_size=64, root='./data'):
    """Returns training and testing dataloaders."""
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = get_serving_transform()

    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    return trainloader, testloader