import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.current_device())  # Should print 0 (or device index)
print(torch.cuda.get_device_name(0))  # Should print GPU name (e.g., "NVIDIA A100")