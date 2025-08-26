#!/bin/bash

# Biosignal Foundation Model - Installation Script
# Supports macOS (Intel/Apple Silicon), Linux, and Windows (WSL)

echo "=================================================="
echo "Biosignal Foundation Model - Installation"
echo "=================================================="

# Detect OS and Architecture
OS=$(uname -s)
ARCH=$(uname -m)

echo "Detected OS: $OS"
echo "Detected Architecture: $ARCH"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch based on platform
echo "Installing PyTorch..."
if [[ "$OS" == "Darwin" ]]; then
    # macOS
    if [[ "$ARCH" == "arm64" ]]; then
        # Apple Silicon (M1/M2)
        echo "Installing PyTorch for Apple Silicon..."
        pip install torch torchvision torchaudio
    else
        # Intel Mac
        echo "Installing PyTorch for Intel Mac..."
        pip install torch torchvision torchaudio
    fi
elif [[ "$OS" == "Linux" ]]; then
    # Linux
    if command -v nvidia-smi &> /dev/null; then
        # CUDA available
        echo "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        # CPU only
        echo "Installing PyTorch CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# Install other requirements
echo "Installing project requirements..."
pip install -r requirements.txt

# Verify installations
echo ""
echo "Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python3 -c "import scipy; print(f'SciPy: {scipy.__version__}')"
python3 -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python3 -c "import wfdb; print(f'WFDB: {wfdb.__version__}')"

# Check GPU availability
echo ""
echo "Checking compute device..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✓ Apple Silicon GPU (MPS) available')
elif torch.cuda.is_available():
    print(f'✓ CUDA GPU available: {torch.cuda.get_device_name(0)}')
else:
    print('⚠ No GPU detected, will use CPU')
"

echo ""
echo "=================================================="
echo "Installation complete!"
echo "=================================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python main.py test --module all"
echo ""