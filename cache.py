import pickle
from os.path import exists


def cache(file):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            if exists(file):
                with open(file, 'rb') as f:
                    return pickle.load(f)

            result = fn(*args, **kwargs)

            with open(file, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

            return result

        return wrapper

    return decorator
