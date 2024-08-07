import sys
import traceback
import json
import time

from contextlib import redirect_stdout


def call_csharp_script(func):
    try:
        start = time.time_ns()
        path_settings = sys.argv[1]
        path_results = sys.argv[2]
        path_log = sys.argv[3]
        durations = {"IO": 0, "Setup": 0, "Actual": 0}

        with open(path_log, 'w') as file_log:
            file_log.write("")
            with redirect_stdout(file_log):
                with open(path_settings) as file_settings:
                    settings = json.load(file_settings)
                    elapsed = (time.time_ns() - start) // 1000000 # in ms
                    durations["IO"] = elapsed
                    func(settings, durations)
        with open(path_results, 'w') as file_results:
            json.dump(durations, file_results)
    except:  # this catches everything, unlike 'except Exception as ex'
        with open(path_log, 'w') as file_log:
            traceback.print_exc(file=file_log)
        sys.exit(100)
    else:
        sys.exit(0)