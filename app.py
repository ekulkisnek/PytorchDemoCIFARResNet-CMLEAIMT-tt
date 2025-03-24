
import streamlit as st
import torch
import torchvision
import matplotlib.pyplot as plt
from model_utils import load_pretrained_model, get_class_labels
from visualization import plot_training_progress, visualize_feature_maps, display_model_architecture
from training import train_model, evaluate_model
import numpy as np

def main():
    # Set up the main title and description for the Streamlit web interface
    # Streamlit is a framework that turns Python scripts into shareable web apps
    st.title("PyTorch Deep Learning Demo: CIFAR-10 Classification")
    st.write("This demo showcases transfer learning using ResNet18 on the CIFAR-10 dataset")

    # Initialize the model and device
    # Check if CUDA (GPU) is available, otherwise use CPU
    # This is crucial for deep learning as GPUs significantly speed up computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model(device)  # Load pre-trained ResNet18 model
    classes = get_class_labels()  # Get CIFAR-10 class names

    # Create sidebar for interactive controls
    # Streamlit sidebar provides a clean way to organize UI controls
    st.sidebar.title("Controls")
    demo_option = st.sidebar.selectbox(
        "Select Demo",
        ["Dataset Exploration", "Model Architecture", "Training", "Feature Visualization", "Inference"]
    )

    # Handle different demo options through conditional rendering
    if demo_option == "Dataset Exploration":
        st.header("CIFAR-10 Dataset Examples")
        
        # Load CIFAR-10 dataset and display random samples
        # ToTensor transform converts PIL images to PyTorch tensors and normalizes to [0,1]
        transform = torchvision.transforms.ToTensor()
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        
        # Create a grid of 5 random images with their class labels
        cols = st.columns(5)
        for i, col in enumerate(cols):
            idx = np.random.randint(len(trainset))  # Get random index
            image, label = trainset[idx]  # Get image and its label
            # Display image with its class label
            # Permute changes tensor dimensions from CxHxW to HxWxC for display
            col.image(image.permute(1, 2, 0).numpy(), caption=f"Class: {classes[label]}")

    elif demo_option == "Model Architecture":
        # Display the model's architecture using string representation
        st.header("ResNet18 Architecture")
        display_model_architecture(model)

    elif demo_option == "Training":
        st.header("Model Training")
        
        # Training hyperparameter controls
        # Allow users to adjust key training parameters
        epochs = st.slider("Number of epochs", 1, 10, 3)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.001, 0.01, 0.1],
            value=0.001,
            format_func=lambda x: f"{x:.4f}"
        )

        # Training execution and progress tracking
        if st.button("Start Training"):
            # Initialize progress tracking components
            progress_bar = st.progress(0)
            status_text = st.empty()
            plot_container = st.empty()
            
            # Start training with real-time progress updates
            # Lambda function updates UI components during training
            history = train_model(model, device, epochs, learning_rate, 
                                progress_callback=lambda epoch, loss, acc: (
                                    progress_bar.progress((epoch + 1) / epochs),
                                    status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {acc:.2f}%"),
                                    plot_training_progress(plot_container, history)
                                ))
            
            st.success("Training completed!")
            
            # Evaluate model on test set after training
            test_accuracy = evaluate_model(model, device)
            st.metric("Test Accuracy", f"{test_accuracy:.2f}%")

    elif demo_option == "Feature Visualization":
        # Visualize feature maps to understand what the model "sees"
        st.header("Feature Map Visualization")
        
        # Load a sample image for visualization
        transform = torchvision.transforms.ToTensor()
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        image, _ = dataset[0]
        
        # Display original image and its feature maps
        st.image(image.permute(1, 2, 0).numpy(), caption="Input Image", width=200)
        visualize_feature_maps(model, image.unsqueeze(0), device)

    elif demo_option == "Inference":
        # Demonstrate model's prediction capabilities
        st.header("Model Inference")
        
        # Load test dataset for inference
        transform = torchvision.transforms.ToTensor()
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        # Display multiple random test images with predictions
        n_samples = 5
        indices = np.random.choice(len(testset), n_samples, replace=False)
        
        cols = st.columns(n_samples)
        for i, col in enumerate(cols):
            image, true_label = testset[indices[i]]
            
            # Get model prediction
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # Disable gradient computation
                output = model(image.unsqueeze(0).to(device))
                pred_label = output.argmax(dim=1).item()
            
            # Display image with true and predicted labels
            col.image(image.permute(1, 2, 0).numpy(), caption=f"True: {classes[true_label]}\nPred: {classes[pred_label]}")

if __name__ == "__main__":
    main()
