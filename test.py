import torch

tensor = torch.rand(4,3,6)

a, b = torch.split(tensor, [3,3], dim=2)

print(tensor.shape, a.shape, b.shape)


tensor = torch.nested.nested_tensor([torch.rand(4,6), torch.rand(3,6), torch.rand(1,6), torch.rand(5,6)])

a, b = torch.split(tensor, [3,3], dim=2)

print([ten.shape for ten in tensor.unbind(0)], "\n", [ten.shape for ten in a.unbind(0)], "\n", [ten.shape for ten in b.unbind(0)])