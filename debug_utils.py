from datetime import datetime

def p(*args, **kwargs):
    """ Print with timestamp """
    time_formated = datetime.now().strftime("%H:%M:%S")
    print(f"\033[91m[{time_formated}] >>>\033[1m", *args, "\033[0m", **kwargs)
