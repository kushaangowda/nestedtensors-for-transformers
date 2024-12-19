from hpml_utils.utils.plots import plot_batch_times
from hpml_utils.utils.torch_utils import save_tensors
from hpml_utils.utils.profiler import profiler

import os
import torch
import random
import argparse
import warnings
from fms.models import get_model
from string import ascii_lowercase
from fms.models.hf import to_hf_api
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pickle

# libraries imported

HUGGINGFACE_MODEL = "amd/AMD-Llama-135m" 
IBM_MODEL_ARCH = "llama"
IBM_MODEL_VARI = "micro"
IBM_MODEL_PATH = "ibm-fms/llama-160m-accelerator"

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = "data"
DATA_PATH = os.path.join(ROOT_PATH, DATA_FOLDER)

NUM_ITERS = 1

# global variables set

# objects defined

def get_filepath(
        filename: str
    ) -> str:
    return os.path.join(DATA_PATH, filename)


def set_seed(
        seed: int = 42, 
        verbose: bool = False
    ) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if verbose:
        print(f"Random seed set as {seed}")


class NestedTensorDataset(Dataset):
    def __init__(
            self, 
            num_samples: int = 200, 
            mode: str = ""
        ) -> Dataset:
        possible_modes = ["nlp", "cv"]
    
        assert mode.lower() in possible_modes, f"please provide one of [{', '.join(possible_modes)}]"  
        
        self.datapoints, self.class_val = [], []

        if mode.lower() == "nlp":
            """ generate random garbage-ish sentences """
            for _ in range(num_samples):
                num_words = random.randrange(10, 256)
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


    def __len__(self) -> int:
        return len(self.datapoints)


    def __getitem__(
            self, 
            idx: int
        ) -> dict:
        assert type(idx) is int, "Integer not provided to retrieve data"
        return {"features": self.datapoints[idx], "labels": self.class_val[idx]}


class NestedTensorCollator():

    def __init__(self, tokenizer, device, max_model_size, is_nest_required):
        self.tokenizer = tokenizer
        self.is_nest_required = is_nest_required
        self.max_model_size = max_model_size
        self.device = device

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
            input_ids = torch.nested.nested_tensor(input_ids, layout=torch.jagged).to(self.device)
            attention_mask = torch.nested.nested_tensor(attention_mask, layout=torch.jagged).to(self.device)

        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            features = self.tokenizer(
                features,
                return_tensors="pt",
                padding=True,
            )
            input_ids = features["input_ids"].remainder(self.max_model_size - 1).to(self.device)
            attention_mask = features["attention_mask"].remainder(self.max_model_size - 1).to(self.device)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# functions and classes defined


def init():
    if not os.path.exists(DATA_PATH): 
        os.makedirs(DATA_PATH)
    else:
        for filename in os.listdir(DATA_PATH):
            file_path = os.path.join(DATA_PATH, filename)
            
            if os.path.isfile(file_path):
                os.remove(file_path)




def main(args, warmup, nest_tensor, seed=555):

    # set global seed 
    set_seed(seed)
    
    profiler.clear()

    # initialize model
    profiler.start("get_model()")
    model = get_model(
        architecture=IBM_MODEL_ARCH, 
        variant=IBM_MODEL_VARI,
        source="hf",
        device_type=args.device,
        norm_eps=1e-6
    )
    profiler.stop("get_model()")


    # convert model to become huggingface compatible 
    profiler.start("to_hf_api()")
    model = to_hf_api(model)
    profiler.stop("to_hf_api()")
 
    profiler.start("tokenizer init")
    tokenizer = AutoTokenizer.from_pretrained(
        IBM_MODEL_PATH,
    )
    profiler.stop("tokenizer init")


    # define dataset
    profiler.start("dataset creation")
    dataset = NestedTensorDataset(
        num_samples=args.num_samples,
        mode=args.mode
    )
    profiler.stop("dataset creation")


    # define data collator
    profiler.start("collator creation")
    collator = NestedTensorCollator(
        tokenizer=tokenizer,
        device=args.device,
        max_model_size=model.config.vocab_size,
        is_nest_required=nest_tensor,
    )
    profiler.stop("collator creation")


    # define dataloader for huggingface trainer
    profiler.start("dataloader creation")
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collator
    )
    profiler.stop("dataloader creation")

    file_name_prefix = f"./data/warmup_{warmup}_nest_tensor_{nest_tensor}"
    
    with open(file_name_prefix+'_init_times.pkl', 'wb') as file:
        pickle.dump(profiler.getAll(), file)
                
    profiler.clear()

    """ WARMUP RUN """
    if warmup:
        for iter in range(1):
            for i, data in enumerate(dataloader):
                input_ids, _, _ = data["input_ids"], data["attention_mask"], data["labels"] 
                output = model(
                    input_ids=input_ids,
                    attention_mask=None,
                )

    profiler.clear()

    """ NORMAL RUN """

    gpu_memory_usage = []
    
    for iter in range(NUM_ITERS):
        
        for i, data in enumerate(dataloader):
            profiler.setPrefix(f"iter {iter + 1} batch {i}")

            profiler.start("inference")
            input_ids, _, _ = data["input_ids"], data["attention_mask"], data["labels"] 
            output = model(
                input_ids=input_ids,
                attention_mask=None,
            )
            gpu_memory_usage.append(torch.cuda.memory_allocated('cuda') / 1e6)
            profiler.stop("inference")
            
            save_tensors(output.logits, "o")
    
    np.save(file_name_prefix+"_gpu.npy", gpu_memory_usage)
    
    with open(file_name_prefix+'_times.pkl', 'wb') as file:
        pickle.dump(profiler.getAll(), file)


if __name__ == "__main__":

    # define and parse cmd args
    parser = argparse.ArgumentParser(description="HPML project group 1")

    parser.add_argument("--num_samples", type=int, help="number of samples in dataset", default=500)
    parser.add_argument("--batch_size", type=int, help="inference batch size", default=32)
    parser.add_argument("--device", type=str, help="computation device", default="cuda")
    parser.add_argument("--mode", type=str, help="dataset mode", default="nlp", choices=['nlp'])
    parser.add_argument("--num_workers", type=int, help="number of dataloader", default=0)
    parser.add_argument('--use_warmup', action='store_true', help='Use warmup')
    parser.add_argument('--use_nested', action='store_true', help='Use nested tensors')
    
    args = parser.parse_args()
            
    warmup = args.use_warmup
    nest_tensor = args.use_nested
            
    print(f">>>> warmup {warmup} nest_tensor {nest_tensor}")
    
    main(args, warmup, nest_tensor, seed=42)