import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import torch
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


def compare_tensors(t1, t2):
    if t1.is_nested or t2.is_nested:
        return all([torch.equal(t1_, t2_) for t1_, t2_ in zip(t1.unbind(0), t2.unbind(0))])
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
        return Fore.GREEN + key1.ljust(15) + " " + key2.ljust(15) + Style.RESET_ALL
    else:
        return Fore.RED + key1.ljust(15) + " " + key2.ljust(15) + Style.RESET_ALL


STATUS = True

for i, (t1, t2) in enumerate(zip(file1.keys(), file2.keys()), 1):
    try:
        # print(i, t1, t2, torch.equal(file1[t1], file2[t2]))
        lengths = get_tensor_lengths(file1[t1])
        tensor1 = truncate_tensor(file2[t2], lengths)
        status = compare_tensors(file1[t1], tensor1)
        print(str(i).ljust(3), check_same_tokens(t1, t2), color_status(status))

        # print only the first instance of unmatched tensors
        if not status and STATUS:
            STATUS = False 
            print(file1[t1], "\n----\n", file2[t2])

    except Exception as err:
        print(f"Tensor Exception occured: {err}")
        print(i, file1[t1], "\n-\n", file2[t2])
    finally:
        # print("="*50)
        pass

