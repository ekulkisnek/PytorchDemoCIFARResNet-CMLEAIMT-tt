
import torch
import torchvision.models as models

def load_pretrained_model(device):
    """Load and prepare the ResNet18 model for CIFAR-10."""
    # Load pre-trained ResNet18 model with ImageNet weights
    # ImageNet is a large dataset of 1000 classes used for pre-training
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Modify the first convolutional layer
    # Original ResNet18 expects 224x224 images, but CIFAR-10 has 32x32 images
    # Change stride from 2 to 1 and kernel size from 7x7 to 3x3
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove max pooling layer
    # Not needed for small CIFAR-10 images, would reduce spatial dimensions too much
    model.maxpool = torch.nn.Identity()
    
    # Modify the final fully connected layer
    # Change output from 1000 (ImageNet classes) to 10 (CIFAR-10 classes)
    num_ftrs = model.fc.in_features  # Get number of input features to final layer
    model.fc = torch.nn.Linear(num_ftrs, 10)  # Replace with new layer for 10 classes
    
    # Move model to specified device (GPU/CPU)
    model = model.to(device)
    return model

def get_class_labels():
    """Return CIFAR-10 class labels."""
    # CIFAR-10 has 10 classes of common objects
    # These labels are used for displaying predictions and true labels
    return ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']
