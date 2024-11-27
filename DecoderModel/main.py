import torch
from torch.nested import nested_tensor
import time
from tqdm import tqdm
import numpy as np
import argparse

from decoderDef import DecoderOnlyModel


def benchmark(batches, position_batches, num_batches, embed_dim, max_seq_len, 
              vocab_size, num_blocks, device, use_nested_tensor):
        
    model_padded = DecoderOnlyModel(
                vocab_size=vocab_size, embed_dim=embed_dim, num_blocks=num_blocks, num_heads=8,
                ff_dim=256, max_seq_len=max_seq_len, use_nested=use_nested_tensor, dropout_p=0.0
            ).to(device)
    
    print("\nWarmup")

    # Warmup
    forward_times = []
    for batch, position_batch in tqdm(zip(batches, position_batches), total=num_batches):
        batch = batch.to(device)
        position_batch = position_batch.to(device)
        torch.cuda.synchronize()
        start_time = time.time()
        output = model_padded(batch, position_batch)
        torch.cuda.synchronize()
        forward_times.append(time.time() - start_time)
    
    print("\nActual Run")

    # Actual run
    forward_times = []
    for batch, position_batch in tqdm(zip(batches, position_batches), total=num_batches):
        batch = batch.to(device)
        position_batch = position_batch.to(device)
        torch.cuda.synchronize()
        start_time = time.time()
        output = model_padded(batch, position_batch)
        torch.cuda.synchronize()
        forward_times.append(time.time() - start_time)
        
    mode = "nested" if use_nested_tensor else "padding"

    print(f"Forward pass time with {mode}: {sum(forward_times)/len(forward_times):.6f} seconds/batch")

    np.save(f"{mode}_{device}_time.npy", np.array(forward_times))
    
    

def getRandomTokens(batch_size, num_batches, max_seq_len, vocab_size):
    
    # generate random token sequences
    seq_lengths = torch.randint(10, max_seq_len + 1, (num_batches, batch_size))
    
    base_tokens = [
        [
            torch.randint(1, vocab_size, (seq_len,)) # 0: padding token, excluded
            for seq_len in batch_seq_lengths
        ]
        for batch_seq_lengths in seq_lengths
    ]
    
    position_ids = [
        [
            torch.arange(seq_len)
            for seq_len in batch_seq_lengths
        ]
        for batch_seq_lengths in seq_lengths
    ]

    
    return base_tokens, position_ids
    
    


def main(
        batch_size, num_batches, max_seq_len, embed_dim, vocab_size, num_blocks,
        device, use_nested_tensor
    ):

    torch.manual_seed(12)
    
    base_tokens, position_ids = getRandomTokens(
                                    batch_size, num_batches, max_seq_len, vocab_size
                                )
    
    layout = torch.jagged if use_nested_tensor else None
    nested_batches = [nested_tensor(batch, layout=layout) for batch in base_tokens]
    nested_position_ids = [nested_tensor(batch, layout=layout) for batch in position_ids]
    
    if use_nested_tensor:
        benchmark(nested_batches, nested_position_ids, num_batches, embed_dim, 
                  max_seq_len, vocab_size, num_blocks, device, use_nested_tensor)
    else:
        padded_batches = [torch.nested.to_padded_tensor(nt, 0.0) for nt in nested_batches]
        padded_position_ids = [
            torch.arange(pb.shape[1]).unsqueeze(0).expand(pb.shape[0], -1)
            for pb in padded_batches
        ]
        
        benchmark(padded_batches, padded_position_ids, num_batches, embed_dim, 
                  max_seq_len, vocab_size, num_blocks, device, use_nested_tensor)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Add arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length.")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension.")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocab size.")
    parser.add_argument("--num_blocks", type=int, default=12, help="Number of decoder blocks.")
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
        args.vocab_size, 
        args.num_blocks, 
        args.device,
        args.use_nested_tensor
    )