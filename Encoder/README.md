## Encoder

Single transformer encoder block

1. `vanillaEncoder.ipynb`  
   A single encoder layer without NestedTensors.  
   Output: `padding_time.npy` (Array of time (in seconds) taken to process each batch).

2. `nestedTensorEncoder.ipynb`  
   A single encoder layer with NestedTensors.  
   Output: `nested_time.npy` (Array of time (in seconds) taken to process each batch).

3. `visualization.ipynb`  
   Uses `padding_time.npy` and `nested_time.npy` to visualize the results.
