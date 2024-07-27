import sys
import traceback
import json

from contextlib import redirect_stdout
from timeit import default_timer as timer


def call_csharp_script(func):
    try:
        start = timer()
        path_settings = sys.argv[1]
        path_results = sys.argv[2]
        durations = {"IO": 0, "setup": 0, "actual": 0}

        with open(path_results, 'w') as file_results:
            file_results.write("")
            with redirect_stdout(file_results):
                with open(path_settings) as file_settings:
                    settings = json.load(file_settings)
                    elapsed = timer() - start
                    durations["IO"] = elapsed
                    func(settings, durations)
        with open(path_settings, 'w') as file_settings:
            json.dump(durations, file_settings)
    except:  # this catches everything, unlike 'except Exception as ex'
        with open(path_results, 'w') as f:
            traceback.print_exc(file=f)
        sys.exit(100)
    else:
        sys.exit(0)