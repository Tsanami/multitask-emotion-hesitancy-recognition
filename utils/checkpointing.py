import json
import os
import dataclasses
import torch


def save_checkpoint_with_config(model, cfg, output_path, seed=None, extra=None):
    """
    Сохраняет веса модели и рядом — JSON со всеми гиперпараметрами конфига.
    Это решает проблему «через месяц непонятно какой ssl_conf_thr использовался».

    output_path: ...../model.pt
    → создаёт также ...../model.config.json
    """
    torch.save(model.state_dict(), output_path)

    cfg_dict = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg.__dict__)
    if seed is not None:
        cfg_dict["_seed"] = seed
    if extra:
        cfg_dict.update(extra)

    config_path = os.path.splitext(output_path)[0] + ".config.json"
    with open(config_path, "w") as f:
        json.dump(cfg_dict, f, indent=4, default=str)
