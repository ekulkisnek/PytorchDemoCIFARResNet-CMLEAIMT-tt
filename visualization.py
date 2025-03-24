
import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def plot_training_progress(container, history):
    """Plot training metrics."""
    # Check if there's data to plot
    if not history['loss']:
        return
    
    # Create subplot figure with two plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    # Loss should decrease over time, indicating model improvement
    ax1.plot(history['loss'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    # Plot training accuracy
    # Accuracy should increase over time
    ax2.plot(history['accuracy'])
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    
    # Display plot in Streamlit container
    container.pyplot(fig)
    plt.close()  # Close figure to free memory

def visualize_feature_maps(model, image, device):
    """Visualize feature maps from the first conv layer."""
    model.eval()  # Set model to evaluation mode
    image = image.to(device)
    
    # Initialize list to store feature maps
    feature_maps = []
    
    # Define hook function to capture output of conv1 layer
    # Hooks allow us to inspect intermediate layer outputs
    def hook(module, input, output):
        feature_maps.append(output)
    
    # Register the hook on conv1 layer
    handle = model.conv1.register_forward_hook(hook)
    
    # Forward pass to get feature maps
    with torch.no_grad():  # Disable gradient computation
        model(image)
    
    # Remove the hook to free memory
    handle.remove()
    
    # Process and display feature maps
    feature_maps = feature_maps[0].cpu()  # Move to CPU and get first batch
    
    # Create 4x4 grid of feature maps
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < 16:  # Display first 16 feature maps
            # Show feature map using viridis colormap
            ax.imshow(feature_maps[0, i].numpy(), cmap='viridis')
        ax.axis('off')  # Hide axes
    
    # Display in Streamlit
    st.pyplot(fig)
    plt.close()

def display_model_architecture(model):
    """Display model architecture summary."""
    # Convert model structure to string and display in code block
    # This shows layers, parameters, and connections
    st.code(str(model))
