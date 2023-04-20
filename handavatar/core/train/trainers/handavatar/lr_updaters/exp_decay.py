
from configs import cfg

def get_customized_lr_names():
    return [k[3:] for k in cfg.train.keys() if k.startswith('lr_')]

def update_lr(optimizer, iter_step, loader_len=None):
    if loader_len is None:
        decay_steps = cfg.train.maxiter #lrate_decay
    else:
        decay_steps = cfg.train.maxepoch * loader_len
    decay_value = cfg.train.lr_decay_rate ** (iter_step / decay_steps)
    for param_group in optimizer.param_groups:
        if f"lr_{param_group['name']}" in cfg.train:
            base_lr = cfg.train[f"lr_{param_group['name']}"]
            new_lrate = base_lr * decay_value
        else:
            new_lrate = cfg.train.lr * decay_value
        param_group['lr'] = new_lrate
