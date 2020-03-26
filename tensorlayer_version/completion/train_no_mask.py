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
network = Unetwork_NMASK
# Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, 'logs\\%s' % network.__name__)
RESULT_PATH = os.path.join(BASE_DIR, 'result\\%s' % network.__name__)
MODEL_PATH = 'checkpoints\\%s' % network.__name__
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def gray2pseudo(gray):
    """
    灰度图转换成伪彩色图
    :param gray: 灰度图
    :return: 伪彩色图
    """
    gray = gray.astype(np.float32)
    gray = ((gray-np.min(gray)) / (np.max(gray)-np.min(gray)))
    sc = np.ones(shape=[gray.shape[0], gray.shape[1], 1], dtype=np.float32) * 1.
    vc = np.ones(shape=[gray.shape[0], gray.shape[1], 1], dtype=np.float32) * 1.
    # gray = gray*60.

    for h in range(gray.shape[0]):
        for w in range(gray.shape[1]):
            v = gray[h, w]
            if v <= 0.55:
                v = v * 110.
                vc[h, w] = vc[h, w]*0.2+0.7
                # v = 0
            # elif 1./3. < v <= 2./3.:
            #     # v = v*180. + 60.
            #     v = 0
            else:
                v = (v-0.5)*220.+150.
            gray[h, w] = v
    gray = np.expand_dims(gray, -1)
    gray = np.concatenate([gray, sc, vc], axis=-1)
    return cv2.cvtColor(gray, cv2.COLOR_HSV2BGR)


def normalize(img):
    img = img.astype(np.float32)
    return (img-np.min(img))/(np.max(img)-np.min(img))


def getDiffMap(raw, depth):
    mask = raw == 0
    mask = mask.astype(np.float32)

    rawMax = np.max(raw)
    raw = raw.astype(np.float32) + mask * rawMax

    diff = np.abs(normalize(raw)-normalize(depth))
    diff = diff*np.abs(mask-1)
    return diff


def validFill(img):
    mask = img == 0
    mask = mask.astype(np.float32)
    rawMax = np.max(img)
    result = img.astype(np.float32) + mask * rawMax
    return result


def validRemove(img, mask_):
    mask = np.expand_dims(mask_, -1)
    mask = np.concatenate([mask, mask, mask], -1)
    mask_r = np.abs(mask - 1)
    result = img * mask_r
    return result


def turnPseduo(depthPath):
    # raw = cv2.imread(depthPath, -1)
    raw = depthPath
    mask = raw == 0
    raw = validFill(raw)
    raw = gray2pseudo(raw)
    raw = validRemove(raw, mask)
    return raw


def train(datapath):
    batch_size = 16

    # create folders to save result images and trained model
    tl.files.exists_or_mkdir(LOG_PATH)
    tl.files.exists_or_mkdir(RESULT_PATH)
    tl.files.exists_or_mkdir(MODEL_PATH)

    # lines = os.listdir(datapath)
    # f = open('C:\\Users\zdj\Desktop\\realsense_list.txt', mode='r')
    # lines = f.readlines()
    # f.close()

    # # ====================== PRE-LOAD DATA =========================== #
    # length = len(lines)
    # train_len = int((length*1.0)//batch_size*batch_size)
    # # train_len = length
    # test_len = int(((length-tra  in_len)//batch_size-1)*batch_size)
    # train_list = lines[:train_len]
    # test_list = lines[train_len:train_len+test_len]

    # ====================== PRE-LOAD DATA =========================== #
    lines = [os.path.join(datapath, path.strip()) for path in os.listdir(datapath) if path.endswith('rawDepth.png')]
    length = len(lines)
    train_len = int((length*0.8)//batch_size*batch_size)
    test_len = int(((length-train_len)//batch_size-1)*batch_size)
    train_list = lines[:train_len]
    test_list = lines[train_len:train_len+test_len]

    # ========================== DEFINE MODEL ============================ #
    # train inference
    img = cv2.imread(train_list[0])
    r, c = img.shape[0], img.shape[1]
    # r, c = 424, 558
    # r, c = 423, 557
    del img
    t_image = tf.placeholder(dtype=tf.float32, shape=[batch_size, r, c, 1], name='t_image_origin_train')
    # t_gray = tf.placeholder(dtype=tf.float32, shape=[batch_size, r, c, 1], name='t_gray')

    # t_mask = tf.placeholder(dtype=tf.float32, shape=[batch_size, r, c, 1], name='t_mask')
    t_target_image = tf.placeholder(dtype=tf.float32, shape=[batch_size, r, c, 1], name='t_target_image')
    # t_image_valid = tf.placeholder(dtype=tf.float32, shape=[1, r, c, 1], name='t_image_valid')

    def refine_net(depth, gray):
        filter_size, stride = 11, 1
        gray_patch = tf.extract_image_patches(gray, ksizes=[1, filter_size, filter_size, 1],
                                              strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
        depth_patch = tf.extract_image_patches(depth, ksizes=[1, filter_size, filter_size, 1],
                                               strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
        # center_depth = depth_patch[:, :, :, filter_size ** 2 // 2:filter_size ** 2 // 2 + 1]
        # mask_depth = tf.sign(tf.abs(depth_patch-center_depth))
        # depth_patch = tf.multiply(depth_patch, mask_depth)

        center = gray_patch[:, :, :, filter_size ** 2 // 2:filter_size ** 2 // 2 + 1]

        w_similar = tf.exp(-tf.div(tf.square(gray_patch-center), 2*6**2))
        w_similar = tf.div(w_similar, tf.reduce_sum(w_similar, axis=-1, keepdims=True))

        gray_patch = tf.depth_to_space(w_similar, filter_size)
        depth_patch = tf.depth_to_space(depth_patch, filter_size)

        depth_patch = tf.multiply(depth_patch, gray_patch)
        name = 'jbf'
        n = InputLayer(depth_patch, name=name + '_input')
        w_mask_init = tf.constant_initializer(value=1.)
        b_mask_init = tf.constant_initializer(value=0.)
        n = Conv2d(n, n_filter=1, filter_size=(filter_size, filter_size), padding='VALID', W_init=w_mask_init,
                   b_init=b_mask_init,
                   strides=(filter_size, filter_size), name=name + '_depth')
        return n
    # net = refine_net(t_image, t_gray)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    learning_rate = 1e-5
    net_g = network(t_image, is_train=True, reuse=False)

    # 定义损失函数
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    tf.summary.scalar('mse_loss', mse_loss)

    # mse_loss_test = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    # tf.summary.scalar('mse_loss_test', mse_loss_test)

    g_vars = tl.layers.get_variables_with_name(network.__name__, True, True)

    # Define optimizer
    g_optim_init = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(mse_loss, var_list=g_vars, global_step=global_step)

    # initialize all viarable
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    tl.layers.initialize_global_variables(sess)

    # Add summary writers
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
    # print('restore success')

    # ========================= train =========================###
    initia_epoch = 10000

    for epoch in range(0, initia_epoch):

        n_iter = 0
        random.shuffle(train_list)
        # shuffle_indices = np.random.permutation(np.arange(train_len))
        # shuffled_list = train_list[shuffle_indices]
        # random.shuffle(train_list)
        # shuffled_label = train_label[shuffle_indices]
        # del shuffled_data, shuffled_label

        for idx in range(0, train_len, batch_size):

            # step_time = time.time()

            b_imgs_list = ['%s-depth.png' % (p.strip().split('-')[0]) for p in train_list[idx: idx + batch_size]]
            b_raw_list = ['%s-rawDepth.png' % (p.strip().split('-')[0]) for p in train_list[idx: idx + batch_size]]
            # b_color_list = ['%s-color.png' % (p.strip().split('-')[0]) for p in train_list[idx: idx + batch_size]]
            # test = b_raw_list[0]
            # order = test.split('\\')[-1].split('-')[0]
            # if order not in L:
            #     continue
            # print(order)

            # if b_raw_list[0].strip().split('\\')[-1].split('-')[0] != '00266':
            #     continue

            # b_imgs_list = ['%s\\%s-depth.png' % (datapath, p.strip()) for p in train_list[idx: idx + batch_size]]
            # b_raw_list = ['%s\\%s-rawDepth.png' % (datapath, p.strip()) for p in train_list[idx: idx + batch_size]]
            # b_color_list = ['%s\\%s-color.png' % (datapath, p.strip()) for p in train_list[idx: idx + batch_size]]
            # print(b_imgs_list[0])
            # b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn)
            # b_color = tl.prepro.threading_data(b_color_list, fn=get_imgs_fn)
            # b_gray = tl.prepro.threading_data(b_color, fn=color2gray)
            b_depths = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn)
            b_rawDepth = tl.prepro.threading_data(b_raw_list, fn=get_imgs_fn)

            # b_gray = np.expand_dims(b_gray, axis=-1)
            b_depths = np.expand_dims(b_depths, axis=-1)
            b_rawDepth = np.expand_dims(b_rawDepth, axis=-1)

            b_depths = b_depths / 1000
            b_rawDepth = b_rawDepth / 1000

            # t1 = datetime.datetime.now()
            # C16, step_ = sess.run([net_g.outputs, global_step], {t_image: b_rawDepth, t_target_image: b_depths})
            _, mse,  step_, summary = sess.run([g_optim_init, mse_loss, global_step, merged_g], {t_image: b_rawDepth, t_target_image: b_depths})
            train_writer.add_summary(summary, step_)

            print('TRAIN: step:%s, mse:%s' % (step_, mse))
            if step_ % 1000 == 0:
                random.shuffle(test_list)
                b_imgs_list = ['%s-depth.png' % (p.strip().split('-')[0]) for p in test_list[: batch_size]]
                b_raw_list = ['%s-rawDepth.png' % (p.strip().split('-')[0]) for p in test_list[: batch_size]]

                b_depths = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn)
                b_rawDepth = tl.prepro.threading_data(b_raw_list, fn=get_imgs_fn)

                b_depths = np.expand_dims(b_depths, axis=-1)
                b_rawDepth = np.expand_dims(b_rawDepth, axis=-1)

                b_depths = b_depths / 1000
                b_rawDepth = b_rawDepth / 1000

                mse, step_, summary = sess.run([mse_loss, global_step, merged_g], {t_image: b_rawDepth, t_target_image: b_depths})
                test_writer.add_summary(summary, step_)
                print('TEST: step:%s, mse:%s' % (step_, mse))

            step_ = sess.run(global_step)
            if step_ % 20000 == 0:

                saver.save(sess, os.path.join(MODEL_PATH, "model_%s.ckpt" % step_))
                print('save success')
            # cv2.imwrite('C:\\Users\zdj\Desktop\\result\\%05s-ours.png' % order, np.squeeze(C16).astype(np.uint16))
            # plt.subplot(121)
            # plt.imshow(C16[0, :, :, 0])
            # plt.show()
            # t2 = datetime.datetime.now()
            # print((t2-t1).microseconds/1e6)

            # t1 = datetime.datetime.now()
            # jbf = sess.run(net.outputs, feed_dict={t_image: C16, t_gray: b_gray})
            # plt.subplot(122)
            # plt.imshow(jbf[0, :, :, 0])
            # plt.show()
            # t2 = datetime.datetime.now()
            # time_ += (t2-t1).microseconds/1e6
            # count += 1
            # print((t2-t1).microseconds/1e6)

            # mask1 = b_rawDepth == 0
            # mask1 = mask1.astype(np.float32)
            # C16 = C16 * mask1 + b_rawDepth
            # count = b_imgs_list[0].strip()
            # rawDepth = turnPseduo(np.squeeze(b_rawDepth).astype(np.uint16))
            # result = turnPseduo(np.squeeze(C16).astype(np.uint16))
            # jbf = turnPseduo(np.squeeze(jbf).astype(np.uint16))
            # cv2.imshow("", np.squeeze(C16).astype(np.uint16))
            # plt.subplot(131), plt.imshow(rawDepth)
            # plt.subplot(132), plt.imshow(result)
            # plt.subplot(133), plt.imshow(jbf)
            # plt.show()
            # cv2.waitKey()
            # cv2.imwrite('C:\\Users\zdj\Desktop\\result\\%05s-result.png' % count, np.squeeze(C16).astype(np.uint16))
            # count += 1
        # print(time_/count)
        # break


def evaluate(datapath):

    def refine_net(depth, gray):
        filter_size, stride = 11, 1
        gray_patch = tf.extract_image_patches(gray, ksizes=[1, filter_size, filter_size, 1],
                                              strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
        depth_patch = tf.extract_image_patches(depth, ksizes=[1, filter_size, filter_size, 1],
                                               strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')

        center = gray_patch[:, :, :, filter_size ** 2 // 2:filter_size ** 2 // 2 + 1]

        w_similar = tf.exp(-tf.div(tf.square(gray_patch-center), 2*6**2))
        w_similar = tf.div(w_similar, tf.reduce_sum(w_similar, axis=-1, keepdims=True))

        gray_patch = tf.depth_to_space(w_similar, filter_size)
        depth_patch = tf.depth_to_space(depth_patch, filter_size)

        depth_patch = tf.multiply(depth_patch, gray_patch)
        name = 'jbf'
        n = InputLayer(depth_patch, name=name + '_input')
        w_mask_init = tf.constant_initializer(value=1.)
        b_mask_init = tf.constant_initializer(value=0.)
        n = Conv2d(n, n_filter=1, filter_size=(filter_size, filter_size), padding='VALID', W_init=w_mask_init,
                   b_init=b_mask_init,
                   strides=(filter_size, filter_size), name=name + '_depth')
        return n

    # ====================== PRE-LOAD DATA =========================== #
    lines = [os.path.join(datapath, path.strip()) for path in os.listdir(datapath) if path.endswith('rawDepth.png')]
    length = len(lines)
    train_len = int((length * 0.8) // batch_size * batch_size)
    test_len = int(((length - train_len) // batch_size - 1) * batch_size)
    train_list = lines[:train_len]
    test_list = lines[train_len:train_len + test_len]

    # ========================== DEFINE MODEL ============================ #
    # train inference
    img = cv2.imread(train_list[0])
    r, c = img.shape[0], img.shape[1]
    # r, c = 424, 558
    # r, c = 423, 557
    del img
    t_image = tf.placeholder(dtype=tf.float32, shape=[batch_size, r, c, 1], name='t_image_origin_test')
    t_gray = tf.placeholder(dtype=tf.float32, shape=[batch_size, r, c, 1], name='t_gray')

    net_g = network(t_image, is_train=True, reuse=False)
    net_g_test = network(t_image, is_train=False, reuse=True)

    # jbf = refine_net(net_g_test.outputs, t_gray)

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
        # saver.restore(sess, save_path=os.path.join(MODEL_PATH, 'model_100.ckpt'))
    print('restore success')

    # f = open('C:\zdj\project\python\RGBD\SR\evaluate\\NYU-RGBD_SR.txt', mode='w')
    total_mse, total_psnr, total_ssim = 0., 0., 0.
    # count = 0
    f = open('C:\\Users\zdj\Desktop\\list.txt', mode='w')
    for idx in range(0, len(lines) - batch_size, batch_size):
        # valid_depths = tl.prepro.threading_data(test_list[idx:idx+batch_size], fn=get_imgs_fn)
        # valid_imgs_640_d = tl.prepro.threading_data(valid_depths, fn=transform_image_depth)
        #
        # out_valid = sess.run(net_g_test.outputs, feed_dict={t_image_valid: valid_imgs_640_d[:1]})
        # out_valid = (out_valid[0, :, :, 0])*3276.8
        #
        # savepath = os.path.join(RESULT_PATH, 'Test-%s-rdb.png' % (test_list[idx:idx+batch_size][0].strip().split('\\')[-1].split('-')[0]))
        # imageio.imwrite(savepath, out_valid.astype(np.uint16))
        # print(test_list[idx:idx+batch_size][0].strip().split('\\')[-1].split('-')[0])
        b_raw_list = ['%s-rawDepth.png' % (p.strip().split('-')[0]) for p in lines[idx: idx + batch_size]]
        b_color_list = ['%s-color.png' % (p.strip().split('-')[0]) for p in lines[idx: idx + batch_size]]
        order = int(lines[idx].strip().split('\\')[-1].split('-')[0])
        print(order)


        b_color = tl.prepro.threading_data(b_color_list, fn=get_imgs_fn)
        b_gray = tl.prepro.threading_data(b_color, fn=color2gray)
        b_rawDepth = tl.prepro.threading_data(b_raw_list, fn=get_imgs_fn)

        b_gray = np.expand_dims(b_gray, axis=-1)
        b_rawDepth = np.expand_dims(b_rawDepth, axis=-1)

        valid_depths = tl.prepro.threading_data(lines[idx:idx + batch_size], fn=get_imgs_fn)
        valid_depths = np.expand_dims(valid_depths, -1)
        # valid_imgs_2560_d = cv2.resize(valid_depths[0], dsize=(valid_depths[0].shape[1] * 4, valid_depths[0].shape[0] * 4),
        #                                interpolation=cv2.INTER_NEAREST)
        # valid_imgs_640_d = tl.prepro.threading_data(valid_depths, fn=transform_image_depth, scale=32768)
        valid_depths = valid_depths / 1000
        output = sess.run(tf.squeeze(net_g_test.outputs * 1000), feed_dict={t_image: valid_depths})
        output = output.astype(np.uint16)
        for i in range(output.shape[0]):
            # plt.imshow(output[i, :, :])
            # plt.show()
            order = lines[idx+i].strip().split('\\')[-1].split('-')[0]
            f.write('%s\n' % order)
            # cv2.imwrite(os.path.join(RESULT_PATH, '%s-nomask.png' % order), output[i, :, :])

        # plt.subplot(241), plt.imshow(output[0, :, :], cmap='gray')
        # plt.subplot(242), plt.imshow(output[1, :, :], cmap='gray')
        # plt.subplot(243), plt.imshow(output[2, :, :], cmap='gray')
        # plt.subplot(244), plt.imshow(output[3, :, :], cmap='gray')
        # plt.subplot(245), plt.imshow(output[4, :, :], cmap='gray')
        # plt.subplot(246), plt.imshow(output[5, :, :], cmap='gray')
        # plt.subplot(247), plt.imshow(output[6, :, :], cmap='gray')
        # plt.subplot(248), plt.imshow(output[7, :, :], cmap='gray')
        # plt.show()
        # cv2.imwrite(os.path.join(RESULT_PATH, '%s-ours.png' % order), output.astype(np.uint16))
        print('finish...')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='evaluate', help='train, evaluate')
    # parser.add_argument('--datapath', type=str, default='RGBD-SCENCES-V1/raw')
    parser.add_argument('--datapath', type=str, default='D:\zdj\Files\Datasets\\NYU\\raw')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['datapath'] = args.datapath
    PATH = ''
    if os.path.exists(tl.global_flag['datapath']):
        PATH = tl.global_flag['datapath']
        if tl.global_flag['mode'] == 'train':
            train(PATH)
        elif tl.global_flag['mode'] == 'evaluate':
            evaluate(PATH)
        # print('1 ' + PATH)
    elif os.path.exists(os.path.join(BASE_DIR, '..\\dataset\\'+tl.global_flag['datapath'])):
        PATH = os.path.join(BASE_DIR, '../dataset/'+tl.global_flag['datapath'])
        # print('2 ' + PATH)
        if tl.global_flag['mode'] == 'train':
            train(PATH)
        elif tl.global_flag['mode'] == 'evaluate':
            evaluate(PATH)
        else:
            raise Exception("Unknow --mode")
    else:
        # print('3' + PATH)
        raise Exception('path do not exit!')
