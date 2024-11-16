import time
from constants import VERBOSE_CODE


class TimeProfiler():
    def __init__(self):
        self._profile_times, self._profile_messages = [], []
        self._lasttime, self._currtime = time.monotonic(), time.monotonic()


    def profile_time(self, message):

        if message.lower() == "start":
            self._lasttime, self._currtime = time.monotonic(), time.monotonic()

        elif message.lower() == "stop":
            self._lasttime, self._currtime = time.monotonic(), time.monotonic()

            assert len(self._profile_times) == len(self._profile_messages), "Somethings wrong with profile times"

            self._profile_messages = list(map(lambda x : "`" + str(x) + "`", self._profile_messages))

            max_length = len(max(self._profile_messages, key=len))
            total_time = sum(self._profile_times)

            if VERBOSE_CODE:
                print("\n" + "="*80, "\nPROFILING TIMES")
                for _time, _msg in zip(self._profile_times, self._profile_messages): 
                    print(f"Time for {_msg.ljust(max_length)}:", f"{_time:08.5f}", "s", "//", f"{((_time*100)/total_time):05.2f}", "%")
                print("="*80)
            self._profile_times, self._profile_messages = [], []

        else:
            self._profile_messages.append(message)
            self._currtime = time.monotonic()
            self._profile_times.append(self._currtime - self._lasttime)

            self._lasttime = self._currtime