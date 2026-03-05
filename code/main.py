import torch


# create a tensor
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# tensor addition
z = x + y

print("PyTorch version:", torch.__version__)
print("Tensor result:", z)