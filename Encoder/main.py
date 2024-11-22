import torch
from torch.nested import nested_tensor
import time
from tqdm import tqdm
import numpy as np
import argparse

from encoderDef import TransformerEncoderWithNested, TransformerEncoderWithPadding

def nested_tensor_benchmark(nested_batches, num_batches, embed_dim, device):

    model_nested = TransformerEncoderWithNested(embed_dim=embed_dim, num_heads=8, ff_dim=256).to(device)

    # Warmup
    forward_times_nested = []
    for nested_batch in tqdm(nested_batches, total=num_batches):
        nested_batch = nested_batch.to(device)
        torch.cuda.synchronize()
        start_time = time.time()
        output = model_nested(nested_batch)
        torch.cuda.synchronize()
        forward_times_nested.append(time.time() - start_time)

    # Actual run
    forward_times_nested = []
    for nested_batch in tqdm(nested_batches, total=num_batches):
        nested_batch = nested_batch.to(device)
        torch.cuda.synchronize()
        start_time = time.time()
        output = model_nested(nested_batch)
        torch.cuda.synchronize()
        forward_times_nested.append(time.time() - start_time)

    print(f"Forward pass time with NestedTensors: {sum(forward_times_nested)/len(forward_times_nested):.6f} seconds/batch")

    np.save("nested_"+device+"_time.npy", np.array(forward_times_nested))
    

def padded_tensor_benchmark(padded_batches, num_batches, embed_dim, device):
        
    model_padded = TransformerEncoderWithPadding(embed_dim=embed_dim, num_heads=8, ff_dim=256).to(device)

    # Warmup
    forward_times_padded = []
    for padded_batch in tqdm(padded_batches, total=num_batches):
        padded_batch = padded_batch.to(device)
        torch.cuda.synchronize()
        start_time = time.time()
        output = model_padded(padded_batch)
        torch.cuda.synchronize()
        forward_times_padded.append(time.time() - start_time)

    # Actual run
    forward_times_padded = []
    for padded_batch in tqdm(padded_batches, total=num_batches):
        padded_batch = padded_batch.to(device)
        torch.cuda.synchronize()
        start_time = time.time()
        output = model_padded(padded_batch)
        torch.cuda.synchronize()
        forward_times_padded.append(time.time() - start_time)

    print(f"Forward pass time with padding: {sum(forward_times_padded)/len(forward_times_padded):.6f} seconds/batch")

    np.save("padding_"+device+"_time.npy", np.array(forward_times_padded))
    


def main(
        batch_size, num_batches, max_seq_len, embed_dim, device, use_nested_tensor
    ):

    torch.manual_seed(12)

    seq_lengths = torch.randint(10, max_seq_len + 1, (num_batches, batch_size))
    base_batches = [
        [torch.randn(seq_len, embed_dim) for seq_len in batch_seq_lengths]
        for batch_seq_lengths in seq_lengths
    ]
    
    layout = torch.jagged if use_nested_tensor else None
    nested_batches = [nested_tensor(batch, layout=layout) for batch in base_batches]
    
    if use_nested_tensor:
        nested_tensor_benchmark(nested_batches, num_batches, embed_dim, device)
    else:
        padded_batches = [torch.nested.to_padded_tensor(nt, 0.0) for nt in nested_batches]
        padded_tensor_benchmark(padded_batches, num_batches, embed_dim, device)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Add arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length.")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda"], help="Device to use for training.")
    parser.add_argument("--use_nested_tensor", action="store_true", help="Use NestedTensor for forward pass optimization.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Pass parsed arguments to the main function
    main(
        args.batch_size,
        args.num_batches,
        args.max_seq_len,
        args.embed_dim,
        args.device,
        args.use_nested_tensor
    )