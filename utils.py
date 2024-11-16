import time
from constants import VERBOSE_CODE


__profile_times, __profile_messages = [], []
__lasttime, __currtime = time.monotonic(), time.monotonic()

def profile_time(message):
    global __lasttime, __currtime, __profile_times, __profile_messages

    if message.lower() == "start":
        __lasttime, __currtime = time.monotonic(), time.monotonic()

    elif message.lower() == "stop":
        __lasttime, __currtime = time.monotonic(), time.monotonic()

        assert len(__profile_times) == len(__profile_messages), "Somethings wrong with profile times"

        __profile_messages = list(map(lambda x : "`" + str(x) + "`", __profile_messages))

        max_length = len(max(__profile_messages, key=len))
        total_time = sum(__profile_times)

        if VERBOSE_CODE:
            print("\n" + "="*80, "\nPROFILING TIMES")
            for _time, _msg in zip(__profile_times, __profile_messages): 
                print(f"Time for {_msg.ljust(max_length)}:", f"{_time:08.5f}", "s", "//", f"{((_time*100)/total_time):05.2f}", "%")
            print("="*80)
        __profile_times, __profile_messages = [], []

    else:
        __profile_messages.append(message)
        __currtime = time.monotonic()
        __profile_times.append(__currtime - __lasttime)

        __lasttime = __currtime