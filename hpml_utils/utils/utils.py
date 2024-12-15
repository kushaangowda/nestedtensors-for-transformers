import time
import inspect

from .constants import COLORS, VERBOSE_CODE

def status_notifier(message):
    print(f"{COLORS["YELLOW"]}{message.upper()}{COLORS["RESET"]}")

class TimeProfiler():

    def __init__(self, verbose=True):
        self.verbose = verbose
        self._profile_times, self._profile_messages = [], []
        self._lasttime, self._currtime = time.monotonic(), time.monotonic()

    def profile_time(self, message):

        if message.lower() == "start":
            self._lasttime, self._currtime = time.monotonic(), time.monotonic()

        elif message.lower() == "stop":
            self._lasttime, self._currtime = time.monotonic(), time.monotonic()

            assert len(self._profile_times) == len(self._profile_messages), "Somethings wrong with profile times"

            max_length = len(max(self._profile_messages, key=len)) + 1
            total_time = sum(self._profile_times)

            if VERBOSE_CODE:
                print("\n" + "="*75, "\nPROFILING TIMES")
                
                for _time, _msg in zip(self._profile_times, self._profile_messages): 
                    print(f"{_msg.ljust(max_length)}:", f"{_time:08.5f}", "s", f"({((_time*100)/total_time):05.2f} %)")
                
                print("="*75)
            self._profile_times, self._profile_messages = [], []

        else:
            if self.verbose: 
                status_notifier(message)
            caller = inspect.stack()[1]
            data = f"{caller.filename.split("/")[-1]} {caller.function}, {message}"
            self._profile_messages.append(data)
            self._currtime = time.monotonic()
            self._profile_times.append(self._currtime - self._lasttime)

            self._lasttime = self._currtime

