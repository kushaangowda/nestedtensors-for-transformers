import os
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.path.expanduser("~"), "hpml_project/nestedtensors-for-transformers", "plots")

def plot_batch_times(
        all_times_1,
        all_times_2,
        all_times_3,
        all_times_4
    ):
    # all_times_1 = [sum(sublist) / len(sublist) for sublist in list(zip(*all_times_1))]
    # all_times_2 = [sum(sublist) / len(sublist) for sublist in list(zip(*all_times_2))]
    # all_times_3 = [sum(sublist) / len(sublist) for sublist in list(zip(*all_times_3))]
    # all_times_4 = [sum(sublist) / len(sublist) for sublist in list(zip(*all_times_4))]

    plt.plot(all_times_1, marker='o', color="blue", label="no-warmup and padded")
    plt.plot(all_times_2, marker='o', color="orange", label="no-warmup and nested")
    plt.plot(all_times_3, marker='o', color="green", label="warmup and padded")
    plt.plot(all_times_4, marker='o', color="red", label="warmup and nested")
    plt.title("All batch GPU Usage")
    plt.xlabel("Batch number")
    plt.ylabel("Memory Usage (MB)")
    plt.xlim(0, len(all_times_1) - 1)
    plt.ylim(0, max([max(all_times_1), max(all_times_2), max(all_times_3), max(all_times_4)]))
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "all_batch_times.png"))
    plt.close()