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
        else:
            models.sort(key=lambda m: '{0:0>15}'.format(m))
            model_name = models[-1]
    else:
        model_name = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(root, model_name)
    return load_path


def get_args():
    custom_parameters = [
        {"name": "--proj_name", "type": str},

        {"name": "--task", "type": str},
        {"name": "--exptid", "type": str},
        {"name": "--resumeid", "type": str},
        {"name": "--checkpoint", "type": int},

        {"name": "--headless", "action": "store_true"},
        {"name": "--simulator", "type": str},
        {"name": "--device", "type": str},
        {"name": "--drive_mode", "type": int},

        {"name": "--debug", "action": "store_true"},
    ]

    # parse arguments
    return parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters
    )


def parse_arguments(description="Example Parser", custom_parameters: list = None):
    parser = argparse.ArgumentParser(description=description)

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"],
                                        help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

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
