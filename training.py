
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

def train_model(model, device, epochs, learning_rate, progress_callback=None):
    """Train the model on CIFAR-10."""
    # Set up data augmentation and normalization transforms
    # Data augmentation helps prevent overfitting by creating variations of training images
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # Convert PIL image to tensor and normalize to [0,1]
        torchvision.transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        torchvision.transforms.RandomCrop(32, padding=4)  # Random crop with padding for more robustness
    ])
    
    # Load CIFAR-10 training dataset
    # download=True automatically downloads the dataset if not present
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    # Create data loader for batch processing
    # Batch processing is more efficient and helps with training stability
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Standard loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer with specified learning rate
    
    # Initialize dictionary to store training history
    history = {'loss': [], 'accuracy': []}
    
    # Training loop
    model.train()  # Set model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Batch training loop
        for i, (inputs, labels) in enumerate(trainloader):
            # Move data to appropriate device (GPU/CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients for each batch
            # This is necessary as PyTorch accumulates gradients
            optimizer.zero_grad()
            
            # Forward pass: compute predictions
            outputs = model(inputs)
            
            # Compute loss between predictions and true labels
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update weights using computed gradients
            optimizer.step()
            
            # Accumulate statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)  # Get the predicted class
            total += labels.size(0)  # Total number of samples
            correct += predicted.eq(labels).sum().item()  # Number of correct predictions
            
        # Calculate epoch metrics
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        
        # Store metrics in history
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback(epoch, epoch_loss, epoch_acc)
    
    return history

def evaluate_model(model, device):
    """Evaluate the model on the test set."""
    # Set up transform for test data (no augmentation needed)
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    
    # Load test dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create test data loader
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )
    
    # Evaluation loop
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)  # Get predicted class
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total  # Return accuracy percentage
