from easydict import EasyDict as edict
import json

config = edict()
config.DATA_PATH = 'D:\Projects\RGB-D_SR\Data'
config.TRAIN = edict()

# Adam
config.TRAIN.batch_size = 1
config.TRAIN.lr_init = 2e-5
config.TRAIN.beta1 = 0.9

# initialize G
config.TRAIN.n_epoch_init = 100
# config.TRAIN.lr_decay_init = 0.1
# config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

# adversarial learning (SRGAN)
config.TRAIN.n_epoch = 4000
config.TRAIN.lr_decay = 0.9
config.TRAIN.decay_every = 4000

# train set location
config.TRAIN.hr_img_path = 'data2017/DIV2K_train_HR/'
config.TRAIN.lr_img_path = 'data2017/DIV2K_train_LR_bicubic/X4/'

config.VALID = edict()
# test set location
config.VALID.hr_img_path = 'data2017/DIV2K_valid_HR/'
config.VALID.lr_img_path = 'data2017/DIV2K_valid_LR_bicubic/X4/'


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
