import os
import pathlib
import torch


def print_seg(diarization):
    for segment, track_name, label in diarization.itertracks(yield_label=True):
        print(f"{segment.start=:.1f}, {segment.end=:.1f}, {track_name=}, {label=}")


def file_list(dir_path):
    return [os.path.join(dir_path, p) for p in os.listdir(dir_path) if pathlib.Path(os.path.join(dir_path, p)).is_file()]


def create_optimizer(model: torch.nn.Module, condition_rate_tuple, momentum=0.1):
    arg_list = []
    model.training = True
    for name, param in model.named_parameters():
        param.requires_grad = False
        for condition, rate in condition_rate_tuple:
            param.requires_grad = True
            if condition(name):
                arg_list.append({'params': param, 'lr': rate})
    return torch.optim.SGD(arg_list, momentum=momentum)
