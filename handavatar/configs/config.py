import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse

import torch

from handavatar.third_parties.yacs import CfgNode as CN

# pylint: disable=redefined-outer-name

_C = CN()

# "resume" should be train options but we lift it up for cmd line convenience
_C.resume = False

# current iteration -- set a very large value for evaluation
_C.eval_iter = 10000000

# for rendering
_C.render_folder_name = ""
_C.ignore_non_rigid_motions = False
_C.render_skip = 1
_C.render_frames = 100

# for data loader
_C.num_workers = 4


def get_cfg_defaults():
    return _C.clone()


def parse_cfg(cfg):
    cfg.logdir = os.path.join('handavatar/out', cfg.category, cfg.task, cfg.subject.replace('/', '_'), cfg.experiment)


def determine_primary_secondary_gpus(cfg):
    print("------------------ GPU Configurations ------------------")
    cfg.n_gpus = torch.cuda.device_count()
    if cfg.n_gpus > 0:
        all_gpus = list(range(cfg.n_gpus))
        cfg.primary_gpus = [0]
        if cfg.n_gpus > 1:
            cfg.secondary_gpus = [g for g in all_gpus]# if g not in cfg.primary_gpus]
        else:
            cfg.secondary_gpus = cfg.primary_gpus
        print(f"Primary GPUs: {cfg.primary_gpus}")
        print(f"Secondary GPUs: {cfg.secondary_gpus}")
    else:
        print(f"CPU job")
    print("--------------------------------------------------------")


def make_cfg(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file('handavatar/configs/default.yaml')
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.file_path = args.cfg
    parse_cfg(cfg)

    determine_primary_secondary_gpus(cfg)
        
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default='handavatar/configs/interhand/test_cap0.yaml', type=str) # 'ROM03_RT_No_Occlusion.yaml'
parser.add_argument("--type", default="freepose", type=str)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

cfg = make_cfg(args)
