import os
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.path.expanduser("~"), "nestedtensors-for-transformers", "plots")

def plot_batch_times(
        all_times_1,
        all_times_2,
        all_times_3,
        all_times_4
    ):
    averages_1 = [sum(sublist) / len(sublist) for sublist in list(zip(*all_times_1))]
    averages_2 = [sum(sublist) / len(sublist) for sublist in list(zip(*all_times_2))]
    averages_3 = [sum(sublist) / len(sublist) for sublist in list(zip(*all_times_3))]
    averages_4 = [sum(sublist) / len(sublist) for sublist in list(zip(*all_times_4))]

    plt.plot(averages_1, marker='o', color="blue", label="no-warmup and padded")
    plt.plot(averages_2, marker='o', color="orange", label="no-warmup and nested")
    plt.plot(averages_3, marker='o', color="green", label="warmup and padded")
    plt.plot(averages_4, marker='o', color="red", label="warmup and nested")
    plt.title("All batch execution times")
    plt.xlabel("Batch number")
    plt.ylabel("Inference time (s)")
    plt.xlim(0, len(averages_1) - 1)
    plt.ylim(0, max([max(averages_1), max(averages_2), max(averages_3), max(averages_4)]))
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "all_batch_times.png"))
    plt.close()
