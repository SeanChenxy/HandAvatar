from configs import cfg
from handavatar.core.utils.log_util import Logger, Board
from handavatar.core.data import create_dataloader
from handavatar.core.nets import create_network
from handavatar.core.train import create_trainer, create_optimizer
import os
import torch


def main():
    log = Logger()
    # log.print_config()
    
    model = create_network()
    phase = cfg.get('phase', 'train')

    if phase=='val':
        trainer = create_trainer(model, None, board=None)
        trainer.progress()
    else:
        board = Board()
        optimizer = create_optimizer(model)
        trainer = create_trainer(model, optimizer, board=board)
        train_loader = create_dataloader('train')
        # estimate start epoch
        epoch = trainer.iter // len(train_loader) + 1
        while True:
            if trainer.iter > cfg.train.maxiter: #cfg.train.maxepoch * len(train_loader):
                break
            
            trainer.train(epoch=epoch,
                        train_dataloader=train_loader)
            epoch += 1

        trainer.finalize()

if __name__ == '__main__':
    main()
