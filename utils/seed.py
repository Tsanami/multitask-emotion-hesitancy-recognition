import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Фиксирует все источники случайности для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Детерминизм cudnn (медленнее, но воспроизводимо)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Для DataLoader(worker_init_fn=seed_worker) при num_workers>0."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
