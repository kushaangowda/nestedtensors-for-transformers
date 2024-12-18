import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import torch
import matplotlib.pyplot as plt
import pickle 

def plot_batch_experiment():
    with open('data/nested_mem_usage.pkl', 'rb') as f:
        nested_dict = pickle.load(f)
        f.close()

    with open('data/vanilla_mem_usage.pkl', 'rb') as f:
        vanilla_dict = pickle.load(f)
        f.close()

    plt.plot(nested_dict['batch_sizes'], nested_dict["gpu"], label = "nested gpu")
    plt.plot(nested_dict['batch_sizes'], nested_dict["cpu"], label = "nested cpu")
    plt.plot(vanilla_dict['batch_sizes'], vanilla_dict["gpu"], label = "vanilla gpu")
    plt.plot(vanilla_dict['batch_sizes'], vanilla_dict["cpu"], label = "vanilla cpu")
    plt.legend()
    plt.savefig("data/batch_plot.png")

if __name__=='__main__':
    plot_batch_experiment()


