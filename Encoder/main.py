import torch
from torch.nested import nested_tensor
import time
from tqdm import tqdm
import numpy as np
import argparse

from encoderDef import TransformerEncoderWithNested, TransformerEncoderWithPadding


def benchmark(batches, num_batches, embed_dim, device, use_nested_tensor):
        
    model_padded = (TransformerEncoderWithNested if use_nested_tensor else TransformerEncoderWithPadding)(
                embed_dim=embed_dim, num_heads=8, ff_dim=256
            ).to(device)
    
    print("\nWarmup")

    # Warmup
    forward_times = []
    for batch in tqdm(batches, total=num_batches):
        batch = batch.to(device)
        torch.cuda.synchronize()
        start_time = time.time()
        output = model_padded(batch)
        torch.cuda.synchronize()
        forward_times.append(time.time() - start_time)
    
    print("\nActual Run")

    # Actual run
    forward_times = []
    for batch in tqdm(batches, total=num_batches):
        batch = batch.to(device)
        torch.cuda.synchronize()
        start_time = time.time()
        output = model_padded(batch)
        torch.cuda.synchronize()
        forward_times.append(time.time() - start_time)
        
    mode = "nested" if use_nested_tensor else "padding"

    print(f"Forward pass time with {mode}: {sum(forward_times)/len(forward_times):.6f} seconds/batch")

    np.save(f"{mode}_{device}_time.npy", np.array(forward_times))
    


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
        benchmark(nested_batches, num_batches, embed_dim, device, use_nested_tensor)
    else:
        padded_batches = [torch.nested.to_padded_tensor(nt, 0.0) for nt in nested_batches]
        benchmark(padded_batches, num_batches, embed_dim, device, use_nested_tensor)
    


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