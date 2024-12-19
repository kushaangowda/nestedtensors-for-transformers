import torch
import time
from collections import defaultdict

class TimeProfiler():
    def __init__(self):
        self.start_times = defaultdict(lambda:[])
        self.end_times = defaultdict(lambda:[])
        self.prefix = ""
        
    def setPrefix(self, prefix):
        self.prefix = prefix
        
    def start(self, msg, cuda_syn=True):
        
        msg = self.prefix + ":" + msg
            
        if cuda_syn:
            torch.cuda.synchronize()
            
        start_time = time.monotonic()
            
        self.start_times[msg].append(start_time)
        
    def stop(self, msg, cuda_syn=True):
        
        msg = self.prefix + ":" + msg
            
        if cuda_syn:
            torch.cuda.synchronize()
            
        end_time = time.monotonic()
            
        self.end_times[msg].append(end_time)
        
    def clear(self):
        self.start_times = defaultdict(lambda:[])
        self.end_times = defaultdict(lambda:[])
        self.prefix = ""
        
    def get(self, msg):
        assert msg in self.start_times and msg in self.end_times
        assert len(self.start_times[msg]) == len(self.start_times[msg])
        
        times = []
        
        for s,e in zip(self.start_times[msg], self.end_times[msg]):
            times.append(e-s)
            
        return times
        
        
    def getAll(self):
        times = {}
        for msg in self.start_times:
            times[msg] = []
            assert msg in self.end_times
            assert len(self.start_times[msg]) == len(self.start_times[msg])
            for s,e in zip(self.start_times[msg], self.end_times[msg]):
                times[msg].append(e-s)
        return times
    
profiler = TimeProfiler()