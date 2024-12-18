from hpml_utils.utils.utils import TimeProfiler
from hpml_utils.utils.plots import plot_batch_times
from hpml_utils.utils.torch_utils import save_tensors

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

import time
import torch.profiler


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

code_profiler = TimeProfiler(verbose=False)
inference_profiler = TimeProfiler(verbose=False)
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
                padding="max_length",
                max_length=100,
                truncation=True,
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
    

    # start program profiling
    code_profiler.profile_time("start")   


    # initialize model
    # model = LlamaForCausalLM.from_pretrained(
    #     HUGGINGFACE_MODEL,
    # )
    model = get_model(
        architecture=IBM_MODEL_ARCH, 
        variant=IBM_MODEL_VARI,
        # model_path=IBM_MODEL_PATH,
        source="hf",
        device_type=args.device,
        norm_eps=1e-6
    )
    code_profiler.profile_time("get_model() called")


    # convert model to become huggingface compatible 
    model = to_hf_api(model)
    code_profiler.profile_time("to_hf_api() called")


    # initialize tokenizer 
    # tokenizer = AutoTokenizer.from_pretrained(
    #     HUGGINGFACE_MODEL,
    #     legacy=False
    # ) 
    tokenizer = AutoTokenizer.from_pretrained(
        IBM_MODEL_PATH,
    )
    code_profiler.profile_time("tokenizer initialized")


    # define dataset
    dataset = NestedTensorDataset(
        num_samples=args.num_samples,
        mode=args.mode
    )
    code_profiler.profile_time("dataset created")


    # define data collator
    collator = NestedTensorCollator(
        tokenizer=tokenizer,
        device=args.device,
        max_model_size=model.config.vocab_size,
        is_nest_required=nest_tensor,
    )
    code_profiler.profile_time("collator loaded")


    # define dataloader for huggingface trainer
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collator
    )
    code_profiler.profile_time("dataset loaded onto generator")


    """ WARMUP RUN """
    if warmup:
        for iter in range(1):
            for i, data in enumerate(dataloader):
                input_ids, attention_mask, _ = data["input_ids"], data["attention_mask"], data["labels"] 
                output = model(
                    input_ids=input_ids,
                    attention_mask=None,
                )
                torch.cuda.synchronize()


    """ NORMAL RUN """

    ''' run 3 runs to take smooth average values '''
    all_batch_times = []
    gpu_memory_usage = []
    for iter in range(NUM_ITERS):

        inference_profiler.profile_time("start")
        inference_times = []
        for i, data in enumerate(dataloader):

            start = time.monotonic()
            input_ids, attention_mask, _ = data["input_ids"], data["attention_mask"], data["labels"] 
            output = model(
                input_ids=input_ids,
                attention_mask=None,
            )
            gpu_memory_usage.append(torch.cuda.memory_allocated('cuda') / 1e6)
            torch.cuda.synchronize()
            print(f"iter {iter + 1}, batch {i} {(time.monotonic()-start):.2f}")
            inference_times.append(time.monotonic()-start)
            inference_profiler.profile_time(f"iter {iter + 1}, batch {i}")

            save_tensors(output.logits, "o")

        inference_profiler.profile_time("stop")

        # inference_times = inference_profiler.get_all_times()
        all_batch_times.append(inference_times)

    # stop profiling    
    code_profiler.profile_time("stop")


    return all_batch_times, gpu_memory_usage


if __name__ == "__main__":
    # init()

    # define and parse cmd args
    parser = argparse.ArgumentParser(description="HPML project group 1")

    parser.add_argument(
        "--num_samples", 
        type=int, 
        help="number of samples in dataset",
        default=500
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="inference batch size",
        default=32
    )

    parser.add_argument(
        "--device", 
        type=str, 
        help="computation device",
        default="cuda"
    )

    parser.add_argument(
        "--mode", 
        type=str, 
        help="dataset mode... can be 'nlp' or 'cv'",
        default="nlp"
    )

    parser.add_argument(
        "--num_workers", 
        type=int, 
        help="number of dataloader",
        default=0
    )

    args = parser.parse_args()

    all_batch_times = []
    gpu_memory_usage = []
    for warmup in [False, True]:
        for nest_tensor in [False, True]:
            print(f">>>> warmup {warmup} nest_tensor {nest_tensor}")
            batch_times, gpu_memory_usage1 = main(args, warmup, nest_tensor, seed=42)
            all_batch_times.append(batch_times)
            gpu_memory_usage.append(gpu_memory_usage1)

    # plot_batch_times(
    #     all_batch_times[0],
    #     all_batch_times[1],
    #     all_batch_times[2],
    #     all_batch_times[3],
    # )

    plot_batch_times(
        gpu_memory_usage[0],
        gpu_memory_usage[1],
        gpu_memory_usage[2],
        gpu_memory_usage[3],
    )