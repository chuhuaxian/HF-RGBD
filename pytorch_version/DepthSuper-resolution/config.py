from easydict import EasyDict as edict

config = edict()

# config.DATAPATH = 'D:\\Projects\\Datasets\\train'
# config.DATAPATH = 'D:\Projects\Datasets\\3dsmax\\evaluate'
config.DATAPATH = 'D:\\Projects\Datasets\\3dsmax\scene'
# config.DATAPATH = 'D:\Projects\Datasets\\3dsmax\\evaluate\\deepshading\Test'

# config.VALDATAPATH = 'Datastes\\Unity_sceenshot\\test1'
config.train_save_dir = 'Logs\\result'
# config.save_name = '9-1-Unet-4'
config.save_name = '01-13-Unet'


# Settings
config.dropout = 0.5
config.learning_rate = 1e-4
config.TRAIN_EPOCH = 100000
config.save_interval = 2000

# config.fc_lst = [128, 64, config.n_classes]

# config.pooling_ratio = 0.8
config.load_model = True
#config.load_model = False
config.model_name = '01-13-Unet\\net_params_25000_2020-01-14_00-28.pkl'

# Hyperparameters
config.epochs = 100000
bs = 4
config.bs = bs

# config.decay_every = int(665600/bs*20)
# print()

config.clip_value = 0.01
config.n_critic = 1