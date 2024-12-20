import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import torch
import matplotlib.pyplot as plt
import os
from colorama import Fore, Style

    
def truncate_tensor(tensor1, tensor2):
    """
    Find the minimum dimensions along all axes 
    and truncate the padded tensor accordingly.
    """
    min_dims = [min(tensor1.size(i), tensor2.size(i)) for i in range(tensor2.ndim)]
    slices = tuple(slice(0, dim) for dim in min_dims)
    return tensor1[slices]
    
def truncate_to_nested(nested, real):

    if not nested.is_nested:
        return nested

    assert nested.is_nested, "tensor is not nested"
    assert not real.is_nested, "tensor is nested"
    
    truncated = []
    for i, target in enumerate(nested.unbind(0)):
        truncated.append(truncate_tensor(real[i], target))

    return torch.nested.nested_tensor(truncated)


def compare_tensors(t1, t2):
    if t1.is_nested or t2.is_nested:
        return all([torch.allclose(t1_, t2_, atol=1e-6, rtol=0) for t1_, t2_ in zip(t1.unbind(0), t2.unbind(0))])
    else:
        return torch.equal(t1, t2)


def truncate_and_match_to_nested(nested, real):
    truncated = truncate_to_nested(nested, real)
    match_status = compare_tensors(nested, truncated)
    tolerance = max([torch.max(abs(t-n)) for (t,n) in zip(truncated, nested)]) 
    return match_status, tolerance.item()


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


def eval(filename1, filename2):
    # Read the outputs of nested and padded tensors
    file1 = torch.load(filename1)
    file2 = torch.load(filename2)

    # Compute tolerance values for each output tensor
    tolerance_vals = []
    for i, (t1, t2) in enumerate(zip(file1.keys(), file2.keys()), 1):
        try:
            status, tolerance = truncate_and_match_to_nested(file1[t1], file2[t2])
            tolerance_vals.append(tolerance)
            
            print(str(i).rjust(3), ">", check_same_tokens(t1, t2), color_status(status), tolerance)

        except Exception as err:
            print(f"Tensor Exception occured with {t1}: ({err})")
        finally:
            pass

    # Create the bar chart
    plt.bar([i for i in range(len(tolerance_vals))], tolerance_vals)

    # Add labels and title
    plt.xlabel('Batch number')
    plt.ylabel('Tolerance Value')
    plt.title('L(inf) norm of Nested - Padded output')

    # Save the chart
    plt.savefig(os.path.join("./plots/output_difference.png"))



if __name__ == "__main__":
    filename1, filename2 = sys.argv[1], sys.argv[2]
    eval(filename1, filename2)

