import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import torch
from print_torch import pt
import matplotlib.pyplot as plt
from colorama import init, Fore, Style

filename1, filename2 = sys.argv[1], sys.argv[2]

file1 = torch.load(filename1)
file2 = torch.load(filename2)

def get_tensor_lengths(tensor):
    if tensor.is_nested:
        return [ele.size(0) for ele in tensor.unbind(0)]
    else:
        return [tensor.size(1) for _ in range(tensor.size(0))]

def truncate_tensor(tensor, size):
    if tensor.ndimension() == 2:
        return torch.nested.nested_tensor([ele[:s_] for ele, s_ in zip(tensor, size)])
    else:
        return torch.nested.nested_tensor([ele[:s_, :] for ele, s_ in zip(tensor, size)])

def print_nested_tensor_shape(nested_tensor):
    assert nested_tensor.is_nested
    
    constituent_shapes = []
    for t in nested_tensor.unbind():
        try:
            shape = tuple(t.size())
        except AttributeError:
            shape = "variable"
        constituent_shapes.append(shape)
    
    print("Nested tensor shapes:")
    for i, shape in enumerate(constituent_shapes):
        print(f"  Tensor {i}: {shape}")

def plot_tensors(t1, t2, name):
    if t1.dim() != 3 or t2.dim() != 3:
        return

    if t1.is_nested:
        t1 = t1.unbind(0)

    if t2.is_nested:
        t2 = t2.unbind(0)

    for i in range(len(t1)):
        plt.imshow(t1[i].cpu().detach().numpy(), cmap='viridis')  # You can change the colormap if you want
        plt.colorbar()  # Add a colorbar to interpret the values
        plt.savefig(f"data/{name}_t1_{i}.png")
        plt.show()
        plt.close()

        plt.imshow(t2[i].cpu().detach().numpy(), cmap='viridis')  # You can change the colormap if you want
        plt.colorbar()  # Add a colorbar to interpret the values
        plt.savefig(f"data/{name}_t2_{i}.png")
        plt.show()
        plt.close()


def truncate_to_nested(nested, real):

    if not nested.is_nested:
        return nested

    assert nested.is_nested, "tensor is not nested"
    assert not real.is_nested, "tensor is nested"
    
    def truncate_tensor(tensor1, tensor2):
        min_dims = [min(tensor1.size(i), tensor2.size(i)) for i in range(tensor2.ndim)]
        slices = tuple(slice(0, dim) for dim in min_dims)
        return tensor1[slices]
    
    truncated = []
    for i, target in enumerate(nested.unbind(0)):
        truncated.append(truncate_tensor(real[i], target))

    return torch.nested.nested_tensor(truncated)

def truncate_and_match_to_nested(nested, real, plot=False):
    truncated = truncate_to_nested(nested, real)
    match_status = compare_tensors(nested, truncated)
    tolerance = max([torch.max(abs(t-n)) for (t,n) in zip(truncated, nested)]) 
    if not match_status and plot:
        plot_tensors(nested, truncated, f"match_fail_{i}")
    return match_status, tolerance


def compare_tensors(t1, t2):
    if t1.is_nested or t2.is_nested:
        return all([torch.allclose(t1_, t2_, atol=2e-6, rtol=0) for t1_, t2_ in zip(t1.unbind(0), t2.unbind(0))])
    else:
        return torch.equal(t1, t2)


def color_status(status):
    if status:
        return Fore.GREEN + str(status) + Style.RESET_ALL
    else:
        return Fore.RED + str(status) + Style.RESET_ALL  


def check_same_tokens(key1, key2):
    similarity = key1.split("-")[0] == key2.split("-")[0] 
    
    if similarity:
        return Fore.GREEN + key1.split("-")[0].ljust(5) + " " + key2.split("-")[0].ljust(5) + Style.RESET_ALL
    else:
        return Fore.RED + key1.split("-")[0].ljust(5) + " " + key2.split("-")[0].ljust(5) + Style.RESET_ALL


def print_nested_tensors_in_one_line(tensor):
    for ele in tensor.unbind(0):
        pt(ele)

MATCH_STATUS = True
for i, (t1, t2) in enumerate(zip(file1.keys(), file2.keys()), 1):
    try:
        status, tolerance = truncate_and_match_to_nested(file1[t1], file2[t2], plot=False)

        print(str(i).rjust(3), ">", check_same_tokens(t1, t2), color_status(status), tolerance)

    except Exception as err:
        print(f"Tensor Exception occured with {t1}: ({err})")
    finally:
        pass

