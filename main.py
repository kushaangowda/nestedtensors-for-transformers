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



# libraries imported

HUGGINGFACE_MODEL = "amd/AMD-Llama-135m" 
IBM_MODEL_ARCH = "llama"
IBM_MODEL_VARI = "micro"
IBM_MODEL_PATH = "/home/harshbenahalkar/llama-160m-accelerator"

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

            input_ids = torch.nested.nested_tensor(input_ids).to(self.device)
            attention_mask = torch.nested.nested_tensor(attention_mask).to(self.device)

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

# basic initialization


def main(filename, nest_flag):
    print("CODE STARTED")

    set_seed(555)
    
    all_input_ids = {}
    all_attention_mask = {}
    all_outputs = {}

    profiler.profile_time("start")   


    # model = LlamaForCausalLM.from_pretrained(
    #     HUGGINGFACE_MODEL,
    # )
    model = get_model(
        architecture=IBM_MODEL_ARCH, 
        variant=IBM_MODEL_VARI,
        # model_path=IBM_MODEL_PATH,
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


    # tokenizer = AutoTokenizer.from_pretrained(
    #     HUGGINGFACE_MODEL,
    #     legacy=False
    # ) 
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

    torch.save(all_input_ids, get_filepath(filename.format(data="inputid")))
    torch.save(all_attention_mask, get_filepath(filename.format(data="attnmask")))
    torch.save(all_outputs, get_filepath(filename.format(data="outputs")))
    profiler.profile_time("stop")
    print("CODE ENDED")

if __name__ == "__main__":
    # init()
    parser = argparse.ArgumentParser(description="HPML project group 1")

    parser.add_argument(
        "--nest_tensors", 
        action="store_true",  # This means the flag will be True if provided, False otherwise.
        help="Enable/Disable nested tensor"
    )
    parser.add_argument(
        "--filepath", 
        type=str, 
        help="Your name",
        default="vanilla_{data}.pt"
    )

    args = parser.parse_args()

    nest_tensors = args.nest_tensors
    filepath = args.filepath
    main(filepath, nest_flag=nest_tensors)