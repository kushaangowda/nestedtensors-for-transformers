## DecoderModel

Decoder only model

1. `main.py`
   ```sh
   python main.py \
      --batch_size=32 \
      --num_batches=50 \
      --max_seq_len=1024 \
      --embed_dim=768 \
      --vocab_size=1000 \
      --num_blocks=12 \
      --device=cuda \
      --use_nested_tensor # Remove this to use padded tensors
   ```
   Main file that runs the benchmark experiment on nested and padded tensors on the GPU.

1. `decoderDef.py`
   File containing the definitions for the decoder model with and without nested tensors.

1. `visualization.ipynb`  
   Uses `padding_cuda_time.npy` and `nested_cuda_time.npy` to visualize the results.
