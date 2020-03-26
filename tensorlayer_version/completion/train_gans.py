#! /usr/bin/python
# -*- coding: utf8 -*-
import time
from completion.models import *
from tools.utils import *
from completion.config import *
import matplotlib.pyplot as plt
import scipy.io as scio
import cv2
import os
import gc
# ====================== HYPER-PARAMETERS =========================== #

# Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
scale = config.TRAIN.scale

# initialize G
n_epoch_init = config.TRAIN.n_epoch_init

# adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

network = Unetwork
network_d = SRGAN_d

# Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, 'logs')
RESULT_PATH = os.path.join(BASE_DIR, 'result')
MODEL_PATH = "checkpoints"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def train(datapath):
    # create folders to save result images and trained model

    tl.files.exists_or_mkdir(LOG_PATH)
    tl.files.exists_or_mkdir(RESULT_PATH)
    tl.files.exists_or_mkdir(MODEL_PATH)

    # ====================== PRE-LOAD DATA =========================== #
    lines = [os.path.join(datapath, path.strip()) for path in os.listdir(datapath) if path.endswith('rawDepth.png')]
    length = len(lines)
    train_len = int((length*0.9)//batch_size*batch_size)
    test_len = int(((length-train_len)//batch_size-1)*batch_size)
    train_list = lines[:train_len]
    test_list = lines[train_len:train_len+test_len]
    # ========================== DEFINE MODEL ============================ #
    # train inference
    img = cv2.imread(train_list[0])
    r, c = img.shape[0], img.shape[1]
    del img
    t_image = tf.placeholder(dtype=tf.float32, shape=[batch_size, r, c, 1], name='t_image_input_to_SRGAN_generator')
    t_image_valid = tf.placeholder(dtype=tf.float32, shape=[1, r, c, 1], name='t_image_valid_input_to_SRGAN_generator')
    t_target_image = tf.placeholder(dtype=tf.float32, shape=[batch_size, r, c, 1], name='t_target_image')

    global_step_g = tf.Variable(0, name="global_step_g", trainable=False)
    global_step_d = tf.Variable(0, name="global_step_d", trainable=False)
    learning_rate = tf.train.exponential_decay(lr_init, global_step_d, decay_every, lr_decay, staircase=True)
    # learn_rate = tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    net_g = network(t_image, is_train=True, reuse=False)
    net_d, logits_real = network_d(t_target_image, is_train=True, reuse=False)
    _, logits_fake = network_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)    # Print all info of parameters in the network

    # test inference
    net_g_test = network(t_image_valid, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)

    total_variation_loss = 0.1*tf.image.total_variation(net_g.outputs)

    g_loss = mse_loss + g_gan_loss+total_variation_loss

    g_vars = tl.layers.get_variables_with_name(network.__name__, True, True)
    d_vars = tl.layers.get_variables_with_name(network_d.__name__, True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    # Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss+total_variation_loss, var_list=g_vars, global_step=global_step_g)
    # SRGAN
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars, global_step=global_step_g)
    d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars, global_step=global_step_d)

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

    # ========================== RESTORE MODEL ============================= #
    # fl = os.listdir(MODEL_PATH)
    # num = []
    # for i in fl:
    #     try:
    #         num.append(int(i.split('.')[0].split('_')[-1]))
    #     except:
    #         continue
    # if len(num) > 0:
    #     saver.restore(sess, save_path=os.path.join(MODEL_PATH, 'model_%s.ckpt' % max(num)))
    # print('restore success， model_%s.ckpt' % max(num))

    # ========================= initialize GAN (SRGAN) =========================###
    initia_epoch = 2000
    for epoch in range(0, initia_epoch):

        n_iter = 0
        random.shuffle(train_list)
        for idx in range(0, len(train_list)-batch_size, batch_size):
            step_time = time.time()

            b_depth_list = ['%s-depth.png' % (p.strip().split('-')[0]) for p in train_list[idx: idx + batch_size]]
            b_raw_list = ['%s-rawDepth.png' % (p.strip().split('-')[0]) for p in train_list[idx: idx + batch_size]]
            # b_color_list = ['%s-color.png' % (p.strip().split('-')[0]) for p in train_list[idx: idx + batch_size]]

            b_depths = tl.prepro.threading_data(b_depth_list, fn=get_imgs_fn)
            b_raws = tl.prepro.threading_data(b_raw_list, fn=get_imgs_fn)

            b_depths = tl.prepro.threading_data(b_depths, fn=transform_image_depth, scale=scale)
            b_raws = tl.prepro.threading_data(b_raws, fn=transform_image_depth, scale=scale)

            # update G
            errM, _, summary, step_ = sess.run([mse_loss, g_optim_init, merged_g, global_step_g],
                                               {t_image: b_raws, t_target_image: b_depths})
            train_writer.add_summary(summary, step_)
            print("Epoch [%2d/%2d] %4d time: %4.4fs, (mse: %.6f)" %
                  (epoch, initia_epoch, n_iter, time.time() - step_time, errM))
            n_iter += 1

            if step_ % 1000 == 0 and step_ != 0:
                random.shuffle(test_list)

                b_depth_list = ['%s-depth.png' % (p.strip().split('-')[0]) for p in test_list[:batch_size]]
                b_raw_list = ['%s-rawDepth.png' % (p.strip().split('-')[0]) for p in test_list[:batch_size]]

                b_depths = tl.prepro.threading_data(b_depth_list, fn=get_imgs_fn)
                b_raws = tl.prepro.threading_data(b_raw_list, fn=get_imgs_fn)

                b_depths = tl.prepro.threading_data(b_depths, fn=transform_image_depth, scale=scale)
                b_raws = tl.prepro.threading_data(b_raws, fn=transform_image_depth, scale=scale)

                ERR_TEST = sess.run(mse_loss, feed_dict={t_image: b_raws, t_target_image: b_depths})
                print("Epoch [%2d/%2d]  time: %4.4fs(mse: %.6f)" %
                      (epoch, initia_epoch,  time.time() - step_time, ERR_TEST))

                out_valid = sess.run(net_g_test.outputs, feed_dict={t_image_valid: b_raws[:1]})
                out_valid = (out_valid[0, :, :, 0])*scale
                savepath = os.path.join(RESULT_PATH, 'Test_step_%s-%s-result.png' %
                                        (step_, b_raw_list[0].strip().split('\\')[-1].split('-')[0]))
                imageio.imwrite(savepath, out_valid.astype(np.uint16))

            #  Save the variables to disk.
            if step_ % 5000 == 0 and step_ != 0:
                # step = sess.run(global_step_g)
                saver.save(sess, os.path.join(MODEL_PATH, "model_%s.ckpt" % step_))
                print('save success')

            if idx % batch_size == 50 and idx != 0:
                del b_depths
                gc.collect()

    # ========================= train GAN (SRGAN) =========================###
    initia_epoch = 2000
    for epoch in range(0, initia_epoch):

        total_d_loss, total_g_loss, n_iter = 0, 0, 0
        epoch_time = time.time()
        random.shuffle(train_list)
        for idx in range(0, len(train_list) - batch_size, batch_size):
            step_time = time.time()

            b_depth_list = ['%s-depth.png' % (p.strip().split('-')[0]) for p in train_list[idx: idx + batch_size]]
            b_raw_list = ['%s-rawDepth.png' % (p.strip().split('-')[0]) for p in train_list[idx: idx + batch_size]]

            b_depths = tl.prepro.threading_data(b_depth_list, fn=get_imgs_fn)
            b_raws = tl.prepro.threading_data(b_raw_list, fn=get_imgs_fn)

            b_depths = tl.prepro.threading_data(b_depths, fn=transform_image_depth, scale=scale)
            b_raws = tl.prepro.threading_data(b_raws, fn=transform_image_depth, scale=scale)

            # update D
            errD, _ = sess.run([d_loss, d_optim], {t_image: b_raws, t_target_image: b_depths})

            # update G
            errG, errM, errA, _, summary, step_ = sess.run([g_loss, mse_loss, g_gan_loss, g_optim, merged_g, global_step_g],
                                                           {t_image: b_raws, t_target_image: b_depths})
            total_d_loss += errD
            total_g_loss += errG
            train_writer.add_summary(summary, step_)
            print("Epoch [%2d/%2d] %4d time: %4.4fs (d_loss: %.8f g_loss: %.8f, mse: %.6f, adv: %.6f)" %
                  (epoch, initia_epoch, n_iter, time.time() - step_time, errD, errG, errM, errA))
            n_iter += 1

            if step_ % 1000 == 0 and step_ != 0:
                random.shuffle(test_list)

                b_depth_list = ['%s-depth.png' % (p.strip().split('-')[0]) for p in test_list[:batch_size]]
                b_raw_list = ['%s-rawDepth.png' % (p.strip().split('-')[0]) for p in test_list[:batch_size]]

                b_depths = tl.prepro.threading_data(b_depth_list, fn=get_imgs_fn)
                b_raws = tl.prepro.threading_data(b_raw_list, fn=get_imgs_fn)

                b_depths = tl.prepro.threading_data(b_depths, fn=transform_image_depth, scale=scale)
                b_raws = tl.prepro.threading_data(b_raws, fn=transform_image_depth, scale=scale)

                ERR_TEST = sess.run(mse_loss, feed_dict={t_image: b_raws, t_target_image: b_depths})
                print("Epoch [%2d/%2d]  time: %4.4fs(mse: %.6f)" % (
                    epoch, initia_epoch, time.time() - step_time, ERR_TEST))

                out_valid = sess.run(net_g_test.outputs, feed_dict={t_image_valid: b_raws[:1]})
                out_valid = (out_valid[0, :, :, 0]) * scale
                savepath = os.path.join(RESULT_PATH, 'Test_step_%s-%s-result.png' %
                                        (step_, b_raw_list[0].strip().split('\\')[-1].split('-')[0]))
                imageio.imwrite(savepath, out_valid.astype(np.uint16))

            if idx % batch_size == 50 and idx != 0:
                del b_depths
                gc.collect()

            #  Save the variables to disk.
            if step_ % 5000 == 0 and step_ != 0:
                saver.save(sess, os.path.join(MODEL_PATH, "model_%s.ckpt" % step_))
                print('save success')

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (
            epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter, total_g_loss / n_iter)
        print(log)


def evaluate(datapath):
    from skimage.measure import compare_ssim
    batch_size  = 1
    t_image_valid = tf.placeholder(tf.float32, [None, 480, 640, 1], name='t_image_input_to_SRGAN_generator')
    # t_image_valid_hr = tf.placeholder(tf.float32, [1, 480*4, 640*4, 1], name='t_image_label')

    net_g = network(t_image_valid, is_train=True, reuse=False)
    net_g_test = network(t_image_valid, is_train=False, reuse=True)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
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
    print('restore success')

    lines = [os.path.join(datapath, path.strip()) for path in os.listdir(datapath) if path.endswith('depth.png')]
    length = len(lines)
    # train_len = int((length*0.8)//batch_size*batch_size)
    # test_len = int(((length-train_len)//batch_size-1)*batch_size)
    # test_list = lines[train_len:train_len+test_len]
    test_list = lines
    f = open('C:\zdj\project\python\RGBD\SR\evaluate\\RGBDV1-SRGAN.txt', mode='w')
    total_mse, total_psnr, total_ssim = 0., 0., 0.
    for idx in range(0, len(test_list)-batch_size, batch_size):
        # valid_depths = tl.prepro.threading_data(test_list[idx:idx+batch_size], fn=get_imgs_fn)
        # valid_imgs_640_d = tl.prepro.threading_data(valid_depths, fn=transform_image_depth)
        #
        # out_valid = sess.run(net_g_test.outputs, feed_dict={t_image_valid: valid_imgs_640_d[:1]})
        # out_valid = (out_valid[0, :, :, 0])*3276.8
        #
        # savepath = os.path.join(RESULT_PATH, 'Test-%s-rdb.png' % (test_list[idx:idx+batch_size][0].strip().split('\\')[-1].split('-')[0]))
        # imageio.imwrite(savepath, out_valid.astype(np.uint16))
        # print(test_list[idx:idx+batch_size][0].strip().split('\\')[-1].split('-')[0])
        valid_depths = tl.prepro.threading_data(test_list[idx:idx+batch_size], fn=get_imgs_fn)
        valid_imgs_2560_d = cv2.resize(valid_depths[0], dsize=(valid_depths[0].shape[1] * 4, valid_depths[0].shape[0] * 4),
                                       interpolation=cv2.INTER_NEAREST)
        valid_imgs_640_d = tl.prepro.threading_data(valid_depths, fn=transform_image_depth)
        output = sess.run(tf.squeeze(net_g_test.outputs * 3276.8), feed_dict={t_image_valid: valid_imgs_640_d[:1]})
        output = output*(255./np.max(output))
        valid_imgs_2560_d = valid_imgs_2560_d.astype(np.float32)
        valid_imgs_2560_d = valid_imgs_2560_d * (255./np.max(valid_imgs_2560_d))

        (score, diff) = compare_ssim(output, valid_imgs_2560_d, full=True)

        MSE = np.mean(np.square(np.abs(output-valid_imgs_2560_d)))
        if math.isnan(MSE):
            length -= 1
            continue
        # mse = sess.run(tl.cost.mean_squared_error((net_g_test.outputs * 3276.8), t_image_valid_hr, is_mean=True),
        #                feed_dict={t_image_valid: valid_imgs_640_d[:1],
        #                           t_image_valid_hr: np.expand_dims(np.expand_dims(valid_imgs_2560_d.astype(np.float32), axis=0), axis=-1)})

        psnr = 10 * np.log(255.**2/MSE)
        total_mse += MSE
        total_psnr += psnr
        total_ssim += score
        print("SSIM: {}, MSE:{}, PSNR:{}".format(score, MSE, psnr))

        f.write("SSIM: %s, MSE: %s, PSNR: %s\n" % (score, MSE, psnr))
        # f.flush()

    f.write('-----------MEAN---------------\n')
    f.write("SSIM: %s, MSE: %s, PSNR: %s" % (total_ssim/length, total_mse/length, total_psnr/length))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, evaluate')
    parser.add_argument('--datapath', type=str, default='NYU/raw')

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
