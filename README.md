# PyTorch Vision Models

A collection of PyTorch implementations for various computer vision models, designed for clarity, efficiency, and ease of use.

## Features
- **Modular Design:** Easily integrate different components (backbones, heads, loss functions) to build custom models.
- **Pre-trained Weights:** Support for loading pre-trained weights for common datasets (e.g., ImageNet).
- **Comprehensive Documentation:** Detailed explanations of model architectures, training procedures, and usage examples.
- **Performance Benchmarks:** Includes scripts for benchmarking model performance on various hardware configurations.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch (latest stable version)
- torchvision
- numpy

### Installation

```bash
git clone https://github.com/Dras1950/pytorch-vision-models.git
cd pytorch-vision-models
pip install -r requirements.txt
```

### Usage

```python
import torch
from models import SimpleCNN

# Initialize model
model = SimpleCNN(num_classes=10)

# Create dummy input
input_tensor = torch.randn(1, 3, 32, 32) # Batch size 1, 3 channels, 32x32 image

# Forward pass
output = model(input_tensor)
print(f"Output shape: {output.shape}")
```

## Models Included
- SimpleCNN: A basic Convolutional Neural Network for image classification.

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
