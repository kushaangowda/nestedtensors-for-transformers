import os
import matplotlib.pyplot as plt
import pickle
import shutil

def getTimes(fileName):
    
    times = {}

    with open(fileName, 'rb') as file:
        data = pickle.load(file)
        
    prefixes = list(set([k.split(":")[0] for k in data.keys()]))
    categories = list(set([k.split(":")[1] for k in data.keys()]))
    
    times = {}
    for category in categories:
        # times[category+" (total)"] = []
        times[category] = []
        for prefix in prefixes:
            total = sum(data[prefix+":"+category])
            # times[category+" (total)"].append(total)
            times[category].append(total/len(data[prefix+":"+category]))
        
    return times


def plot(a1, a2, a3, a4, title, xlabel, ylabel):
    # plt.plot(a2, marker='o', color="orange", label="no-warmup and nested")
    plt.plot(a4, marker='o', color="red", label="nested")
    # plt.plot(a1, marker='o', color="blue", label="no-warmup and padded")
    plt.plot(a3, marker='o', color="green", label="padded")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join("./plots/"+title+".png"))
    plt.close()
    
def get_time_plot():
    times_00 = getTimes('./data/warmup_False_nest_tensor_False_times.pkl')
    times_01 = getTimes('./data/warmup_False_nest_tensor_True_times.pkl')
    times_10 = getTimes('./data/warmup_True_nest_tensor_False_times.pkl')
    times_11 = getTimes('./data/warmup_True_nest_tensor_True_times.pkl')
    
    if os.path.exists("./plots"):
        shutil.rmtree("./plots")
        
    os.mkdir("./plots/")

    for k in times_00:
        plot(
            times_00[k],
            times_01[k],
            times_10[k],
            times_11[k],
            k,
            "Batch number",
            "Time (s)"
        )
    
    
if __name__ == "__main__":
    get_time_plot()