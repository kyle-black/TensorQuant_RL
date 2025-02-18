
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())  # Should return True if CUDA is available