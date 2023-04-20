import sys
import os

from termcolor import colored
from configs import cfg

from tensorboardX import SummaryWriter
import torch

class Logger(object):
    r"""Duplicates all stdout to a file."""
    def __init__(self):
        path = os.path.join(cfg.logdir, 'logs.txt')

        log_dir = cfg.logdir
        # if not cfg.resume and os.path.exists(log_dir):
        #     user_input = input(f"log dir \"{log_dir}\" exists. \nContinue? (y/n):")
        #     if user_input == 'y':
        #         print(colored('continue to save contents in directory %s' % log_dir, 'red'))
        #         # print(colored('remove contents of directory %s' % log_dir, 'red'))
        #         # os.system('rm -r %s/*' % log_dir)
        #     else:
        #         print(colored('exit from the training.', 'red'))
        #         exit(0)

        if not os.path.exists(log_dir):
            os.makedirs(cfg.logdir)

        self.log = open(path, "a") if os.path.exists(path) else open(path, "w")
        self.stdout = sys.stdout
        sys.stdout = self

        os.system("""cp -r {0} "{1}" """.format(cfg.file_path, os.path.join(cfg.logdir, 'conf.yaml')))


    def write(self, message):
        self.stdout.write(message)
        self.stdout.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

    def print_config(self):
        print("\n\n######################### CONFIG #########################\n")
        print(cfg)
        print("\n##########################################################\n\n")


class Board(object):
    def __init__(self):
        path = os.path.join(cfg.logdir, 'board')
        os.makedirs(path, exist_ok=True)

        self.board = SummaryWriter(path)
    
    def board_scalar(self, phase, n_iter, lr=None, **kwargs):
        split = '/'
        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, torch.Tensor):
                        val = val.item()
                    self.board.add_scalar(phase + split + key + split + sub_key, sub_val, n_iter)
            elif isinstance(val, tuple):
                for sub_key, sub_val in enumerate(val):
                    self.board.add_scalar(phase + split + key + split + str(sub_key), sub_val, n_iter)
            else:
                self.board.add_scalar(phase + split + key, val, n_iter)
        if lr:
            self.board.add_scalar(phase + split + 'lr', lr, n_iter)
    
    def board_img(self, phase, n_iter, data):

        self.board.add_image(phase, data.transpose(2, 0, 1), n_iter)
