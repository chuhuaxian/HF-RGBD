from easydict import EasyDict as edict

config = edict()

# config.DATAPATH = 'D:\\Projects\\Datasets\\train'
# config.DATAPATH = 'D:\Projects\Datasets\\3dsmax\\evaluate'
config.DATAPATH = 'D:\\Projects\Datasets\\3dsmax\scene'
# config.DATAPATH = 'D:\Projects\Datasets\\3dsmax\\evaluate\\deepshading\Test'

# config.VALDATAPATH = 'Datastes\\Unity_sceenshot\\test1'
config.train_save_dir = 'Logs\\result'
# config.save_name = '9-1-Unet-4'
config.save_name = '02-09-Unet'
# config.save_name = '02-16-Unet'


# Settings
config.dropout = 0.5
config.learning_rate = 1e-5
config.TRAIN_EPOCH = 100000
config.save_interval = 10000

# config.fc_lst = [128, 64, config.n_classes]

# config.pooling_ratio = 0.8
config.load_model = True
# config.load_model = False
config.model_name = '%s\\net_params_614000_2020-02-13_03-01.pkl' % config.save_name
# config.model_name = '%s\\net_params_1390000_2020-02-26_10-58.pkl' % config.save_name


# Hyperparameters
config.epochs = 100000
bs = 2
config.bs = bs

# config.decay_every = int(665600/bs*20)
# print()

config.clip_value = 0.01
config.n_critic = 1