import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from hpml_utils.utils.utils import TimeProfiler, status_notifier
from hpml_utils.utils.torch_utils import get_percentage_zero, get_all_lengths, save_tensors

import os
import argparse
import random
from string import ascii_lowercase
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, AutoTokenizer
from fms.models import get_model
from fms.models.hf import to_hf_api

import torch.profiler
import pandas as pd
import matplotlib.pyplot as plt
import pickle 

# libraries imported

HUGGINGFACE_MODEL = "amd/AMD-Llama-135m" 
IBM_MODEL_ARCH = "llama"
IBM_MODEL_VARI = "micro"
IBM_MODEL_PATH = "ibm-fms/llama-160m-accelerator"

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = "data"
DATA_PATH = os.path.join(ROOT_PATH, DATA_FOLDER)

# global variables set

profiler = TimeProfiler(verbose=False)

# objects defined

def get_filepath(filename):
    return os.path.join(DATA_PATH, filename)

def set_seed(seed: int = 42):
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


class NestedTensorDataset(Dataset):
    def __init__(self, num_samples=200, mode=""):
        possible_modes = ["nlp", "cv"]
    
        assert mode.lower() in possible_modes, f"please provide one of [{', '.join(possible_modes)}]"  
        
        self.datapoints, self.class_val = [], []

        if mode.lower() == "nlp":
            """ generate random garbage-ish sentences """
            for _ in range(num_samples):
                num_words = random.randrange(4, 22)
                temp = []
                for _ in range(num_words):
                    word_length = random.randrange(1, 10)
                    word = ''.join(random.choices(ascii_lowercase, k=word_length))
                    temp.append(word)
                
                self.datapoints.append(" ".join(temp))

                random_class = random.randrange(0, 10)
                self.class_val.append(random_class)
        else:
            raise NotImplementedError("Please implement the CV part as well")

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        assert type(idx) is int, "Integer not provided to retrieve data"
        return {"features": self.datapoints[idx], "labels": self.class_val[idx]}

class NestedTensorCollator():

    def __init__(self, tokenizer, device, max_model_size, is_nest_required, max_seq_len=100):
        self.tokenizer = tokenizer
        self.is_nest_required = is_nest_required
        self.max_model_size = max_model_size
        self.device = device
        self.max_seq_len = max_seq_len

    def __call__(self, examples):
        """ tokenize string data and then nest it """
        features = list(map(lambda x : x["features"], examples))
        labels = torch.tensor(list(map(lambda x : x["labels"], examples)))

        if self.is_nest_required:
            features = self.tokenizer(
                features,
                return_tensors=None,
                padding=False,
                truncation=False,
            )
            input_ids, attention_mask = [], []
            for input_id, a_mask in zip(features["input_ids"], features["attention_mask"]):
                input_ids.append(torch.tensor(input_id).remainder(self.max_model_size - 1))
                attention_mask.append(torch.tensor(a_mask).remainder(self.max_model_size - 1))

            input_ids = torch.nested.nested_tensor(input_ids).to(self.device)
            attention_mask = torch.nested.nested_tensor(attention_mask).to(self.device)

        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            features = self.tokenizer(
                features,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
            )
            input_ids = features["input_ids"].remainder(self.max_model_size - 1).to(self.device)
            attention_mask = features["attention_mask"].remainder(self.max_model_size - 1).to(self.device)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# functions and classes defined

# basic initialization
def init():
    if not os.path.exists(DATA_PATH): 
        os.makedirs(DATA_PATH)
    set_seed(555)

def get_cuda_allocated_memory():
    return torch.cuda.memory_allocated() / 1000000

def get_max_memory_usage(profiler):
    sorted_results = sorted(
        list(profiler.key_averages()), 
        key=lambda x: x.cuda_memory_usage, 
        reverse=True  # This sorts from highest to lowest
    )
    max_gpu = sorted_results[0].cuda_memory_usage / 1000000

    sorted_results = sorted(
        list(profiler.key_averages()), 
        key=lambda x: x.cpu_memory_usage, 
        reverse=True  # This sorts from highest to lowest
    )
    max_cpu = sorted_results[0].cpu_memory_usage / 1000000

    # Accessing specific columns or performing operations
    # print(data_df.head())  # View first few rows
    # print(data_df.columns)  # See available columns
    # print(profiler.key_averages().table(sort_by="cuda_memory_usage", row_limit=1))

    return max_gpu, max_cpu

def profile_memory_run(filename, batch_size=4, device="cuda", nest_flag=False):
    torch.cuda.empty_cache()

    model = get_model(
        architecture=IBM_MODEL_ARCH, 
        variant=IBM_MODEL_VARI,
        source="hf",
        device_type=device,
        norm_eps=1e-6
    )
    model = to_hf_api(model)

    tokenizer = AutoTokenizer.from_pretrained(
        IBM_MODEL_PATH,
    )

    dataset = NestedTensorDataset(
        num_samples=200,
        mode="nlp"
    )

    all_batch_mems = []
    
    collator = NestedTensorCollator(
        tokenizer=tokenizer,
        device=device,
        max_model_size=model.config.vocab_size,
        is_nest_required=nest_flag,
        max_seq_len=100,
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collator
    )

    with torch.profiler.profile(profile_memory=True) as prof:
        for i, data in enumerate(dataloader):
            input_ids, attention_mask, labels = data["input_ids"], data["attention_mask"], data["labels"] 
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            break

    return get_max_memory_usage(prof)        

def profile_memory(filename, nest_flag):
    # Batch size
    mem_usage = {
        "gpu": [],
        "cpu": []
    }
    batch_sizes = [2, 4, 8, 16, 32]
    for batch_size in batch_sizes:
        g, c = profile_memory_run(filename, batch_size=batch_size, nest_flag=nest_flag)
        mem_usage["gpu"].append(g)
        mem_usage["cpu"].append(c)
    mem_usage["batch_sizes"] = batch_sizes

    with open(get_filepath(filename.format(data='mem_usage.pkl')), 'wb') as f:
        pickle.dump(mem_usage, f)
        f.close()

    # Max Sequence Length


def main(filename, nest_flag):
    print("CODE STARTED")
    
    all_input_ids = {}
    all_attention_mask = {}
    all_outputs = {}

    profiler.profile_time("start")   

    model = get_model(
        architecture=IBM_MODEL_ARCH, 
        variant=IBM_MODEL_VARI,
        source="hf",
        device_type="cuda",
        norm_eps=1e-6
    )
    profiler.profile_time("get_model() called")

    model = to_hf_api(model)
    profiler.profile_time("to_hf_api() called")

    model_vocab_size = model.config.vocab_size
    print(f"Model Vocabulary Size: {model_vocab_size}")
    profiler.profile_time("max vocab size determined")

    tokenizer = AutoTokenizer.from_pretrained(
        IBM_MODEL_PATH,
    )
    profiler.profile_time("tokenizer initialized")


    dataset = NestedTensorDataset(
        num_samples=200,
        mode="nlp"
    )
    profiler.profile_time("dataset created")


    collator = NestedTensorCollator(
        tokenizer=tokenizer,
        device="cuda",
        max_model_size=model_vocab_size,
        is_nest_required=nest_flag,
    )
    profiler.profile_time("collator loaded")


    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collator
    )
    profiler.profile_time("dataset loaded onto generator")

    for i, data in enumerate(dataloader):
        print(f"batch {i} STEPS > ", end="")
        input_ids, attention_mask, labels = data["input_ids"], data["attention_mask"], data["labels"] 
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        all_input_ids[i] = input_ids
        all_attention_mask[i] = attention_mask
        all_outputs[i] = output.logits

        save_tensors(output.logits, "o")
        break
    profiler.profile_time("dataset processed")

    torch.save(all_input_ids, get_filepath(filename.format(data="inputid")))
    torch.save(all_attention_mask, get_filepath(filename.format(data="attnmask")))
    torch.save(all_outputs, get_filepath(filename.format(data="outputs")))
    profiler.profile_time("stop")
    print("CODE ENDED")

def read_args():
    parser = argparse.ArgumentParser(description="HPML project group 1")
    parser.add_argument(
        "--nest_tensors", 
        action="store_true", 
        help="Enable/Disable nested tensor"
    )
    parser.add_argument(
        "--filepath", 
        type=str, 
        help="Filepath for output",
        default="vanilla_{data}"
    )
    parser.add_argument(
        "--mem", 
        action="store_true", 
        help="Perform memory profiling",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    init()
    args = read_args()

    if args.mem:
        profile_memory(args.filepath, nest_flag=args.nest_tensors)
    else:
        main(args.filepath, nest_flag=args.nest_tensors)