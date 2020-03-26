import os
import random, time
import numpy as np
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from SR.models import RGBD_SR, RDBs_Network, RGBD_SR_NO_MASK

from tools.utils import *
from SR.config import config, log_config
import matplotlib.pyplot as plt
import scipy.io as scio
import imageio
import gc
import cv2

# ====================== HYPER-PARAMETERS =========================== #

# Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

# initialize G
n_epoch_init = config.TRAIN.n_epoch_init

# adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

# ni = int(np.sqrt(batch_size))
network = RGBD_SR
# network = RGBD_SR_NO_MASK
# network = RDBs_Network
# Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, 'logs\\%s' % network.__name__)
RESULT_PATH = os.path.join(BASE_DIR, 'result\\%s' % network.__name__)
MODEL_PATH = 'checkpoints\\%s' % network.__name__
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def train(datapath):
    # create folders to save result images and trained model

    tl.files.exists_or_mkdir(LOG_PATH)
    tl.files.exists_or_mkdir(RESULT_PATH)
    tl.files.exists_or_mkdir(MODEL_PATH)

    # ====================== PRE-LOAD DATA =========================== #
    lines = [os.path.join(datapath, path.strip()) for path in os.listdir(datapath) if path.endswith('-depth.png')]
    length = len(lines)
    train_len = int((length*0.8)//batch_size*batch_size)
    test_len = int(((length-train_len)//batch_size-1)*batch_size)
    train_list = lines[:train_len]
    test_list = lines[train_len:train_len+test_len]


    # ========================== DEFINE MODEL ============================ #
    img = cv2.imread(train_list[0])
    r, c = img.shape[0], img.shape[1]
    del img
    # train inference
    t_image = tf.placeholder(dtype=tf.float32, shape=[batch_size, r//4, c//4, 1], name='t_image_input_to_SRGAN_generator')
    t_image_valid = tf.placeholder(dtype=tf.float32, shape=[1, r, c, 1], name='t_image_valid_input_to_SRGAN_generator')
    t_target_image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 420, 556, 1], name='t_target_image')

    global_step = tf.Variable(0, name="global_step", trainable=False)
    learning_rate = tf.train.exponential_decay(lr_init, global_step, decay_every, lr_decay, staircase=True)
    # learn_rate = tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    net_g = network(t_image, is_train=True, reuse=False)
    net_g_test = network(t_image, is_train=False, reuse=True)
    net_g_valid = network(t_image_valid, is_train=False,  reuse=True)

    net_g.print_params(False)    # Print all info of parameters in the network

    # 定义损失函数
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    tf.summary.scalar('mse_loss', mse_loss)

    total_variation_loss = 0.1 * tf.image.total_variation(net_g.outputs)

    total_loss = mse_loss
    # g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # Pretrain
    g_optim_init = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(total_loss, var_list=g_vars, global_step=global_step)

    # initialize all viarable
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    tl.layers.initialize_global_variables(sess)

    # Add summary writers
    # merged_g = tf.summary.merge([mse_loss_summary, g_loss_summary, learn_rate])
    merged_g = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOG_PATH, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOG_PATH, 'test'))

    s_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(s_vars)
    # print('===============', tf.get_collection(tf.GraphKeys.VARIABLES))
    # ========================== RESTORE MODEL ============================= #
    # saver.restore(sess, save_path=os.path.join(MODEL_PATH, 'model_9095.ckpt'))
    # print('restore success')
    # ========================= initialize GAN (SRGAN) =========================###
    initia_epoch = 10000
    for epoch in range(0, initia_epoch):

        n_iter = 0
        random.shuffle(train_list)

        for idx in range(0, len(train_list)-batch_size, batch_size):
            step_time = time.time()
            b_imgs_list = train_list[idx: idx + batch_size]

            b_depths = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn)


            b_imgs_640_d = tl.prepro.threading_data(b_depths, fn=transform_image_depth, scale=1000.)
            b_imgs_640_d = b_imgs_640_d[:, 3:, 1:, :]
            b_imgs_160_d = tl.prepro.threading_data(b_imgs_640_d, fn=subsample_depth)

            # update G

            errM, _, summary, step_ = sess.run([mse_loss, g_optim_init, merged_g, global_step],
                                               {t_image: b_imgs_160_d,
                                                t_target_image: b_imgs_640_d})
            train_writer.add_summary(summary, step_)
            print("Epoch [%2d/%2d] %4d time: %4.4fs(mse: %.6f)" % (epoch, initia_epoch, n_iter, time.time() - step_time, errM))
            n_iter += 1

            if step_ % 2000 == 0:
                step = sess.run(global_step)
                saver.save(sess, os.path.join(MODEL_PATH, "model_%s.ckpt" % step))
                print('save success')
                # batch_size = 1
                for idx in range(0, len(lines), 1):
                    # b_imgs_list = lines[idx: idx + batch_size]
                    #
                    # b_depths = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn)
                    #
                    # b_imgs_640_d = tl.prepro.threading_data(b_depths, fn=transform_image_depth, scale=1000.)
                # random.shuffle(test_list)
                # b_imgs_list = test_list[:batch_size]
                # b_depths = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn)
                # b_imgs_640_d = tl.prepro.threading_data(b_depths, fn=transform_image_depth, scale=1000.)
                #
                # b_imgs_160_d = tl.prepro.threading_data(b_imgs_640_d, fn=subsample_depth)
                # test_list = test_list[:batch_size]
                    test_list = lines[idx:idx+1]
                    valid_depths = tl.prepro.threading_data(test_list, fn=get_imgs_fn)
                    valid_imgs_640_d = tl.prepro.threading_data(valid_depths, fn=transform_image_depth, scale=1000.)

                # ERR_TEST = sess.run(mse_loss, feed_dict={t_image: b_imgs_160_d, t_target_image: b_imgs_640_d})
                # print(ERR_TEST)
                # print("Epoch [%2d/%2d]  time: %4.4fs(mse: %.6f)" % (
                # epoch, initia_epoch,  time.time() - step_time, ERR_TEST))
                    out_valid = sess.run(net_g_valid.outputs, feed_dict={t_image_valid: valid_imgs_640_d})
                    out_valid = out_valid[0, :, :, 0] * 1000.
                # out_test = (out_test[0, :, :, 0])*32768.
                #     out_valid = (out_valid[0, :, :, 0])*1000.
                #     for i in range(batch_size):
                    savepath = os.path.join(RESULT_PATH, 'step_%s_%s-result.png' %
                                            (step, test_list[0].strip().split('\\')[-1].split('-')[0]))
                    imageio.imwrite(savepath, out_valid.astype(np.uint16))



            if idx % batch_size == 50 and idx != 0:
                del b_depths, b_imgs_640_d, b_imgs_160_d
                gc.collect()
            #     # Save the variables to disk.
            # if step_ % 100 == 0 and step_ != 0:
            #     step = sess.run(global_step)
            #     saver.save(sess, os.path.join(MODEL_PATH, "model_%s.ckpt" % step))
            #     print('save success')


def evaluate(datapath):
    from skimage.measure import compare_ssim
    batch_size = 1
    lines = [os.path.join(datapath, path.strip()) for path in os.listdir(datapath) if path.endswith('-depth.png')]
    length = len(lines)
    train_len = int((length*0.8)//batch_size*batch_size)
    test_len = int(((length-train_len)//batch_size-1)*batch_size)
    train_list = lines[:train_len]
    test_list = lines[train_len:train_len+test_len]

    # batch_size  = 1
    img = cv2.imread(lines[0])
    r, c = img.shape[0], img.shape[1]
    del img
    t_image_valid = tf.placeholder('float32', [batch_size, r, c, 1], name='t_image_input_to_SRGAN_generator')
    # t_image_valid_hr = tf.placeholder(tf.float32, [1, r*4, c*4, 1], name='t_image_label')

    net_g = network(t_image_valid, is_train=True, reuse=False)
    net_g_test = network(t_image_valid, is_train=False, reuse=True)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    s_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(s_vars)

    # ========================== RESTORE MODEL ============================= #
    fl = os.listdir(MODEL_PATH)
    num = []
    for i in fl:
        try:
            num.append(int(i.split('.')[0].split('_')[-1]))
        except:
            continue
    if len(num) > 0:
        saver.restore(sess, save_path=os.path.join(MODEL_PATH, 'model_%s.ckpt' % max(num)))
        # print(max(num))
        # print('')
        # saver.restore(sess, save_path=os.path.join(MODEL_PATH, 'model_100.ckpt'))
    print('restore success')

    # f = open('C:\zdj\project\python\RGBD\SR\evaluate\\NYU-RGBD_SR.txt', mode='w')
    total_mse, total_psnr, total_ssim = 0., 0., 0.
    for idx in range(0, len(lines), batch_size):
        # valid_depths = tl.prepro.threading_data(test_list[idx:idx+batch_size], fn=get_imgs_fn)
        # valid_imgs_640_d = tl.prepro.threading_data(valid_depths, fn=transform_image_depth)
        #
        # out_valid = sess.run(net_g_test.outputs, feed_dict={t_image_valid: valid_imgs_640_d[:1]})
        # out_valid = (out_valid[0, :, :, 0])*3276.8
        #
        # savepath = os.path.join(RESULT_PATH, 'Test-%s-rdb.png' % (test_list[idx:idx+batch_size][0].strip().split('\\')[-1].split('-')[0]))
        # imageio.imwrite(savepath, out_valid.astype(np.uint16))
        # print(test_list[idx:idx+batch_size][0].strip().split('\\')[-1].split('-')[0])
        order = lines[idx].strip().split('\\')[-1].split('-')[0]
        print(order)
        if order != '10124':
            continue
        valid_depths = tl.prepro.threading_data(lines[idx:idx+batch_size], fn=get_imgs_fn)
        # valid_imgs_2560_d = cv2.resize(valid_depths[0], dsize=(valid_depths[0].shape[1] * 4, valid_depths[0].shape[0] * 4),
        #                                interpolation=cv2.INTER_NEAREST)
        valid_imgs_640_d = tl.prepro.threading_data(valid_depths, fn=transform_image_depth, scale=32678.)
        output = sess.run(tf.squeeze(net_g_test.outputs * 32678.), feed_dict={t_image_valid: valid_imgs_640_d})
        order = lines[idx].strip().split('\\')[-1].split('-')[0]
        output = output.astype(np.uint16)
        # f.write('%s\n' % order)
        cv2.imwrite(os.path.join(RESULT_PATH, '%s-ours_hr.png' % order), output)
        # for i in range(output.shape[0]):
            # plt.imshow(output[i, :, :])
            # plt.show()

        # plt.imshow(output)
        # plt.show()
        # cv2.imwrite(os.path.join(RESULT_PATH, '%s-ours.png' % order), output.astype(np.uint16))
        # print('')
        # output = cv2.resize(valid_depths[0].astype(np.float32), dsize=(valid_depths[0].shape[1] * 4, valid_depths[0].shape[0] * 4),
        #                                interpolation=cv2.INTER_CUBIC)
        # output = output.astype(np.float32)

        # RMSE = np.sqrt(np.mean(np.square(np.abs(output/1000.-valid_imgs_2560_d/1000.))))
        #
        # output = output*(255./np.max(output))
        # valid_imgs_2560_d = valid_imgs_2560_d.astype(np.float32)
        # valid_imgs_2560_d = valid_imgs_2560_d * (255./np.max(valid_imgs_2560_d))
        #
        # # (score, diff) = compare_ssim(output, valid_imgs_2560_d, full=True)
        #
        # # mse = sess.run(tl.cost.mean_squared_error((net_g_test.outputs * 3276.8), t_image_valid_hr, is_mean=True),
        # #                feed_dict={t_image_valid: valid_imgs_640_d[:1],
        # #                           t_image_valid_hr: np.expand_dims(np.expand_dims(valid_imgs_2560_d.astype(np.float32), axis=0), axis=-1)})
        #
        # MSE = np.mean(np.square(np.abs(output - valid_imgs_2560_d)))
        # psnr = 10 * np.log(255.**2/MSE)
        #
        # if math.isnan(RMSE) or RMSE == 0 or math.isnan(psnr):
        #     length -= 1
        #     continue
        #
        # total_mse += RMSE
        # total_psnr += psnr
        # # total_ssim += score
        # print("RMSE:{}, PSNR:{}".format(RMSE, psnr))
        #
        # f.write("MSE: %s, PSNR: %s\n" % (RMSE, psnr))
        # f.flush()

    # f.write('-----------MEAN---------------\n')
    # f.write("MSE: %s, PSNR: %s" % (total_mse/length, total_psnr/length))
        # print(MSE, '  ----------  ', psnr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='evaluate', help='train, evaluate')
    parser.add_argument('--datapath', type=str, default='NYU/raw1')
    # parser.add_argument('--datapath', type=str, default='RGBD-SCENCES-V2/raw')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['datapath'] = args.datapath
    PATH = ''
    if os.path.exists(tl.global_flag['datapath']):
        PATH = tl.global_flag['datapath']
    elif os.path.exists(os.path.join(BASE_DIR, '..\\dataset\\'+tl.global_flag['datapath'])):
        PATH = os.path.join(BASE_DIR, '../dataset/'+tl.global_flag['datapath'])
        if tl.global_flag['mode'] == 'train':
            train(PATH)
        elif tl.global_flag['mode'] == 'evaluate':
            evaluate(PATH)
        else:
            raise Exception("Unknow --mode")
    else:
        raise Exception('path do not exit!')
