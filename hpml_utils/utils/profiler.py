import torch
import time
from collections import defaultdict

class TimeProfiler():
    """
    A utility for profiling execution time.

    Methods:
    - setPrefix(prefix: str): Sets the prefix for the time tracking strings.
    - start(label: str): Starts the timer to track the time associated with the given label name.
    - stop(label: str): Stops the timer and records the elapsed time for the given label name.
    - clear(): Clears all recorded timing data.
    - get(label: str): Retrieves all recorded times associated with the given label name.
    - getAll(): Retrieves all recorded timing data.
    
    Example: prefix can be "Batch 10" and label can be "Attention"
    """

    def __init__(self):
        self.start_times = defaultdict(lambda:[])
        self.end_times = defaultdict(lambda:[])
        self.prefix = ""
        
    def setPrefix(self, prefix):
        self.prefix = prefix
        
    def start(self, label, cuda_sync=True):
        
        label = self.prefix + ":" + label
            
        if cuda_sync:
            torch.cuda.synchronize()
            
        start_time = time.monotonic()
            
        self.start_times[label].append(start_time)
        
    def stop(self, label, cuda_sync=True):
        
        label = self.prefix + ":" + label
            
        if cuda_sync:
            torch.cuda.synchronize()
            
        end_time = time.monotonic()
            
        self.end_times[label].append(end_time)
        
    def clear(self):
        self.start_times = defaultdict(lambda:[])
        self.end_times = defaultdict(lambda:[])
        self.prefix = ""
        
    def get(self, label):
        assert label in self.start_times and label in self.end_times
        assert len(self.start_times[label]) == len(self.start_times[label])
        
        times = []
        
        for s,e in zip(self.start_times[label], self.end_times[label]):
            times.append(e-s)
            
        return times
        
        
    def getAll(self):
        times = {}
        for label in self.start_times:
            times[label] = []
            assert label in self.end_times
            assert len(self.start_times[label]) == len(self.start_times[label])
            for s,e in zip(self.start_times[label], self.end_times[label]):
                times[label].append(e-s)
        return times
    
profiler = TimeProfiler()