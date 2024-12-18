import torch


tensor = torch.nested.as_nested_tensor([torch.rand(4,6), torch.rand(3,6), torch.rand(1,6), torch.rand(5,6)])

print(tensor)

print("\\")

print(tensor + tensor)