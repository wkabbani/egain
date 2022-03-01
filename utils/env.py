# What version of Python do you have?
import sys

import torch
from torchsummary import summary
from models.encoders import psp_encoders

print(f"PyTorch Version: {torch.__version__}")
print(f"Python {sys.version}")
print("GPU is", "available" if torch.cuda.is_available() else "NOT AVAILABLE")
