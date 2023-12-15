import numpy as np           
import time

def set_see(seed_value: int = 42) -> None:
    np.random.seed(seed_value)


def get_unique_filename(filename, ext):
    return time.strftime(f"{filename}_%Y_%m_%d_%H_%M.{ext}")


print(get_unique_filename("model", "pth"))  # model_name ,extension
