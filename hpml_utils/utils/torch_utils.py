import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
import uuid

def get_all_lengths(tensors):
    """ gives all the dimensions in a nested tensor of dim 2"""

    def get_dim(ten):
        return ",".join([str(dim) for dim in ten.size()])

    if tensors.is_nested:
        shapes = []
        for tensor in tensors.unbind():
            shapes.append(get_dim(tensor))

        shapes = list(map(lambda x : f"<{x}>", shapes))
        print(tensors.size(0), "->", " ".join(shapes))

    else:
        shapes = ["<" + get_dim(tensors[0]) + ">" for _ in range(tensors.size(0))]
        print(tensors.size(0), "->", " ".join(shapes))



def get_percentage_zero(tensors):
    total_elements, zero_count = [], []
    for tensor in tensors.unbind(0):
        total = int(torch.numel(tensor))

        non_zero_count = int(torch.count_nonzero(tensor))

        total_elements.append(total)
        zero_count.append(total - non_zero_count)

    print(">>", sum(zero_count), "zero elements out of", sum(total_elements))


def save_tensors(tensor, key=None):

    filepath = os.environ.get("LOGFILE", None)

    if filepath is None:
        raise ValueError("Unable to get filepath, check again")
        
    if os.path.exists(filepath):
        data = torch.load(filepath)
    
    else:
        data = {}

    unique_id = str(uuid.uuid4())[:8] 

    key = key + "-" if key is not None else ""
    key = key + unique_id
        
    data[key] = tensor
    torch.save(data, filepath)

