import argparse
import os
import random

import numpy as np
import torch


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_load_path(root, checkpoint, model_name_starts_with="model"):
    if not os.path.isdir(root):
        raise ValueError(f"resume directory \"{root}\" is not a directory")

    if checkpoint is None:
        models = [file for file in os.listdir(root) if file.startswith(model_name_starts_with)]

        if 'latest.pt' in os.listdir(root):
            model_name = 'latest.pt'
        elif len(models) > 0:
            models.sort(key=lambda m: '{0:0>15}'.format(m))
            model_name = models[-1]
        else:
            raise ValueError(f"No checkpoint found at \"{root}\"")

    else:
        model_name = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(root, model_name)
    return load_path


def get_args():
    parser = argparse.ArgumentParser(description="BridgeDP RL framework")
    parser.add_argument("--proj_name", type=str, required=True)

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--exptid", type=str, required=True)
    parser.add_argument("--resumeid", type=str)
    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--headless", action="store_true")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--simulator", type=int, help="0 for IsaacGym, 1 for Genesis")
    parser.add_argument("--drive_mode", type=int, help="0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort")

    args = parser.parse_args()
    _, _, args.device = parse_device_str(args.device)
    return args


def parse_device_str(device_str: str):
    assert device_str.startswith('cpu') or device_str.startswith('gpu') or device_str.startswith('cuda'), f'Invalid device string "{device_str}"'

    if device_str == 'cpu' or device_str == 'cuda':
        device = device_str
        device_id = 0

        if device_str == 'cuda':
            device_str = 'cuda:0'
    else:
        device_args = device_str.split(':')
        device, device_id_s = device_args
        try:
            device_id = int(device_id_s)
        except ValueError:
            raise ValueError(f'Invalid device string "{device_str}". Cannot parse "{device_id_s}"" as a valid device id')
    return device, device_id, device_str
