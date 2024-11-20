from utils import TimeProfiler
import random
from string import ascii_lowercase
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, AutoTokenizer

torch.manual_seed(42)


"""
TASKS

// create a dataset generator
// create a dataloader
create a datacollator
bitsandbytes config
complete loading of model 

"""


HUGGINGFACE_MODEL = "amd/AMD-Llama-135m" 
profiler = TimeProfiler()

class NestedTensorDataset(Dataset):
    def __init__(self, num_samples=10, mode=""):
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
                    sentence = ''.join(random.choices(ascii_lowercase, k=word_length))
                    temp.append(sentence)
                
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
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def __call__(self, examples):
        """ tokenize string data and then nest it """
        features = list(map(lambda x : x["features"], examples))
        labels = list(map(lambda x : x["labels"], examples))
        
        features = self.tokenizer(
            features,
        )
        input_ids = torch.nested.nested_tensor(features["input_ids"])
        attention_mask = torch.nested.nested_tensor(features["attention_mask"])

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
 

if __name__ == "__main__":
    profiler.profile_time("start")    
    model = LlamaForCausalLM.from_pretrained(
        HUGGINGFACE_MODEL,
    )
    profiler.profile_time("model initialized")

    tokenizer = AutoTokenizer.from_pretrained(
        HUGGINGFACE_MODEL,
        legacy=False
    )
    profiler.profile_time("tokenizer initialized")

    dataset = NestedTensorDataset(
        num_samples=10,
        mode="nlp"
    )
    profiler.profile_time("dataset created")

    collator = NestedTensorCollator(
        tokenizer=tokenizer
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
        print("LOOP", i)
        # print(data)
        input_ids, attention_mask, labels = data["input_ids"], data["attention_mask"], data["labels"] 
        # output = model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask
        # )
        # print(output)
        # print("")
        # for input_id in input_ids:
            # print("id>>", input_id) 
        
        print("")

    profiler.profile_time("stop")