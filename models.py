import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.

    Architecture:
    - Two convolutional layers with ReLU activation and max pooling.
    - Two fully connected layers.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        # Input: 3 channels (RGB images), Output: 32 channels, Kernel size: 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Second convolutional layer
        # Input: 32 channels, Output: 64 channels, Kernel size: 3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # The input size to the first fully connected layer depends on the input image size
        # For a 32x32 input image, after two conv+pool layers, the feature map size will be 8x8
        # (32 / 2 / 2 = 8). So, 64 channels * 8 * 8 = 4096 features.
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        # -> conv1 -> relu -> pool
        x = self.pool(F.relu(self.conv1(x)))
        # -> conv2 -> relu -> pool
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 8 * 8)

        # -> fc1 -> relu
        x = F.relu(self.fc1(x))
        # -> fc2 (output layer)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Example usage:
    # Create a dummy input tensor (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 32, 32)

    # Instantiate the model
    model = SimpleCNN(num_classes=10)

    # Perform a forward pass
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model architecture:")
    print(model)

    # Verify the number of parameters (optional)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
