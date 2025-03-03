import random
import torch
import numpy as np#获取
#获取GPU_id标识，使用cuda训练
def get_device(device_id) -> torch.device:
    return torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

#结果保存的基础路径
def data_path() -> str:
    return '/data0/data_wk/'


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './ruslt/'

def checkpoint_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './checkpoint/'

def tsne_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './tsne/'
def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
