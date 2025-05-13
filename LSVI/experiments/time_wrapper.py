import time
from functools import wraps


def timer(runs=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            for _ in range(runs):
                start_time = time.perf_counter()
                func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            return times

        return wrapper

    return decorator
