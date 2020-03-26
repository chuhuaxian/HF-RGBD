import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np


class Network(object):

    all_layers = []
    all_params = []
    all_drop = {}

    def __init__(self, name, input, label):
        self.name = name
        self.input = input
        self.label = label

    def forward(self):
        pass

    def get_loss(self):
        pass

    def optimize(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass


# w_init = tf.random_normal_initializer(mean=1., stddev=0.02)
w_init = tf.contrib.layers.xavier_initializer()
b_init = None  # tf.constant_initializer(value=0.0)


def P_Conv2(net, mask, n_filter=32, filter_size=3, stride=1, name=''):

    img_patch = tf.extract_image_patches(net.outputs, ksizes=[1, filter_size, filter_size, 1],
                                         strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
    img_patch = tf.depth_to_space(img_patch, filter_size)
    img_patch = tf.multiply(img_patch, mask)
    n = InputLayer(img_patch, name=name+'_input')
    n = Conv2d(n, n_filter=n_filter, filter_size=(filter_size, filter_size), padding='VALID', W_init=w_init, b_init=b_init, strides=(filter_size, filter_size), name=name+'_depth')

    return n


def get_mask(img, filter_size=3):
    # threshold = tf.Variable(initial_value=500/32768., dtype=tf.float32)
    threshold = 500/32768.
    img_patch = tf.extract_image_patches(img, ksizes=[1, filter_size, filter_size, 1],
                                         strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    temp = img_patch[:, :, :, filter_size**2//2:filter_size**2//2+1]
    img_patch = tf.sign(tf.nn.relu(-(tf.abs(img_patch-temp)-threshold)))
    count = tf.reduce_sum(img_patch, axis=-1, keepdims=True)
    img_patch = tf.div(img_patch, count)
    img_patch = tf.depth_to_space(img_patch, filter_size)
    return img_patch


def RGBD_SR(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    times = 4
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        # nearst = tl.prepro.threading_data(t_image, fn=tf.image.resize_images, size=(t_image.shape[1]*4, t_image.shape[2]*4), method=0)
        # nearst = tf.image.resize_images(t_image, (16, t_image.shape[1]*4, t_image.shape[2]*4, 1), method=0)
        # b_size, height, width, c = t_image.get_shape().as_list()
        # m = tf.Variable(tf.random_normal([b_size, height*3, width*3]), name="conv1_weights",dtype=tf.float32)
        n = InputLayer(t_image, name='in')

        mask = get_mask(t_image)
        # n = Conv2d(n, 32*times, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        n = P_Conv2(n, mask, n_filter=32*times, name='pconv1')

        # n = Conv2d(n, 32, (1, 1), (1, 1), act=tf.identity, W_init=w_init, name='I1')
        # n = Conv2d(n, 256, (1, 1), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='out')
        # n.outputs = tf.multiply(n.outputs, mask)
        temp = n
        #
        # # B residual blocks
        for i in range(2):
            # nn = Conv2d(n, 32*times, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = P_Conv2(n, mask, n_filter=32 * times, name='n64s1/c1/%s' % i)
            # nn = InstanceNormLayer(nn, act=tf.nn.relu, name='istn%s_1'%i)
            # nn = Conv2d(nn, 32, (1, 1), (1, 1), act=tf.identity, W_init=w_init, name='I2')
            # nn.outputs = tf.multiply(nn.outputs, mask)
            # nn = BatchNormLayer(nn, act=tf.nn.relu, gamma_init=g_init, is_train=is_train,  name='n64s1/b1/%s' % i)
            # nn = Conv2d(nn, 32*times, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = P_Conv2(nn, mask, n_filter=32 * times, name='n64s1/c2/%s' % i)
            # nn = InstanceNormLayer(nn, act=tf.nn.relu, name='istn%s_2' % i)
            # nn = Conv2d(nn, 32, (1, 1), (1, 1), act=tf.identity, W_init=w_init, name='I3')
            # nn.outputs = tf.multiply(nn.outputs, mask)
            # nn = BatchNormLayer(nn, act=tf.nn.relu, gamma_init=g_init, is_train=is_train, name='n64s1/b2/%s' % i)
            # nn = ElementwiseLayer([n, nn], combine_fn=tf.add, name='b_residual_add/%s' % i)
            nn = ConcatLayer([n, nn], name='b_residual_add/%s' % i)
            n = nn
        #
        # # n = Conv2d(n, 32*times, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = P_Conv2(n, mask, n_filter=32*times, name='n64s1/c/m')
        # n = InstanceNormLayer(n, act=tf.nn.relu, name='istn')
        # # n = Conv2d(n, 32, (1, 1), (1, 1), act=tf.identity, W_init=w_init, name='I4')
        # # n.outputs = tf.multiply(n.outputs, mask)
        # n = BatchNormLayer(n, gamma_init=g_init, is_train=is_train,  name='n64s1/b/m')
        # n = ElementwiseLayer([n, temp], combine_fn=tf.add,  name='concat2')
        n = ConcatLayer([n, temp], name='concat2')
        # B residual blacks end

        # n = Conv2d(n, 64*times, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = P_Conv2(n, mask, n_filter=64 * times, name='n256s1/1')
        # n = P_Conv1(n, m, filter_size=1, n_filter=256, name='n256s1/1')
        # n = Conv2d(n, 256, (1, 1), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='out')
        # n = Conv2d(n, , (1, 1), (1, 1), act=tf.identity, W_init=w_init, name='I5')
        # n.outputs = tf.multiply(n.outputs, mask)
        n = SubpixelConv2d(n, scale=4, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        # n = Conv2d(n, 64*times, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        # m = UpSampling2dLayer(m, [2, 2], True, 1, name='upsample15_c')
        # n = P_Conv1(n, m, n_filter=64 * times, name='n256s1/2')
        # n = Conv2d(n, 64, (1, 1), (1, 1), act=tf.identity, W_init=w_init, name='I6')
        # n.outputs = tf.multiply(n.outputs, tf.image.resize_images(mask, size=[mask.shape[1]*2, mask.shape[2]*2], method=0))
        # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')
        # n = InstanceNormLayer(n)
        # n = ConcatLayer([n, UpSampling2dLayer(origin, [4, 4], True, 1, name='Upsample_out')], name='concat3')
        # n1 = UpSampling2dLayer(origin, [4, 4], True, 1, name='Upsample_out')
        # n1 = InstanceNormLayer(n1, name='innorm')
        # n = ConcatLayer([n, n1], name='concat5')
        n = Conv2d(n, 1, (1, 1), (1, 1), act=tf.nn.relu, padding='VALID', W_init=w_init, name='out')

        # n.outputs = tf.multiply(n.outputs, tf.image.resize_images(mask, [mask.shape[1] * 4, mask.shape[2] * 4], method=0))
        # n.outputs = tf.multiply(n.outputs, mask)
        return n


def RGBD_SR_NO_MASK(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    times = 4
    with tf.variable_scope(RGBD_SR_NO_MASK.__name__, reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        # nearst = tl.prepro.threading_data(t_image, fn=tf.image.resize_images, size=(t_image.shape[1]*4, t_image.shape[2]*4), method=0)
        # nearst = tf.image.resize_images(t_image, (16, t_image.shape[1]*4, t_image.shape[2]*4, 1), method=0)
        # b_size, height, width, c = t_image.get_shape().as_list()
        # m = tf.Variable(tf.random_normal([b_size, height*3, width*3]), name="conv1_weights",dtype=tf.float32)
        n = InputLayer(t_image, name='in')
        DenseLayer
        # mask = get_mask(t_image)
        # n = Conv2d(n, 32*times, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        # n = P_Conv2(n, mask, n_filter=32*times, name='pconv1')
        n = Conv2d(n,  n_filter=32 * times, name='pconv1')

        # n = Conv2d(n, 32, (1, 1), (1, 1), act=tf.identity, W_init=w_init, name='I1')
        # n = Conv2d(n, 256, (1, 1), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='out')
        # n.outputs = tf.multiply(n.outputs, mask)
        temp = n
        #
        # # B residual blocks
        for i in range(2):
            # nn = Conv2d(n, 32*times, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = Conv2d(n,  n_filter=32 * times, name='n64s1/c1/%s' % i)
            # nn = InstanceNormLayer(nn, act=tf.nn.relu, name='istn%s_1'%i)
            # nn = Conv2d(nn, 32, (1, 1), (1, 1), act=tf.identity, W_init=w_init, name='I2')
            # nn.outputs = tf.multiply(nn.outputs, mask)
            # nn = BatchNormLayer(nn, act=tf.nn.relu, gamma_init=g_init, is_train=is_train,  name='n64s1/b1/%s' % i)
            # nn = Conv2d(nn, 32*times, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = Conv2d(nn,  n_filter=32 * times, name='n64s1/c2/%s' % i)
            # nn = InstanceNormLayer(nn, act=tf.nn.relu, name='istn%s_2' % i)
            # nn = Conv2d(nn, 32, (1, 1), (1, 1), act=tf.identity, W_init=w_init, name='I3')
            # nn.outputs = tf.multiply(nn.outputs, mask)
            # nn = BatchNormLayer(nn, act=tf.nn.relu, gamma_init=g_init, is_train=is_train, name='n64s1/b2/%s' % i)
            # nn = ElementwiseLayer([n, nn], combine_fn=tf.add, name='b_residual_add/%s' % i)
            nn = ConcatLayer([n, nn], name='b_residual_add/%s' % i)
            n = nn
        #
        # # n = Conv2d(n, 32*times, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = Conv2d(n,  n_filter=32*times, name='n64s1/c/m')
        # n = InstanceNormLayer(n, act=tf.nn.relu, name='istn')
        # # n = Conv2d(n, 32, (1, 1), (1, 1), act=tf.identity, W_init=w_init, name='I4')
        # # n.outputs = tf.multiply(n.outputs, mask)
        # n = BatchNormLayer(n, gamma_init=g_init, is_train=is_train,  name='n64s1/b/m')
        # n = ElementwiseLayer([n, temp], combine_fn=tf.add,  name='concat2')
        n = ConcatLayer([n, temp], name='concat2')
        # B residual blacks end

        # n = Conv2d(n, 64*times, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = Conv2d(n,  n_filter=64 * times, name='n256s1/1')
        MaxPool2d
        # n = P_Conv1(n, m, filter_size=1, n_filter=256, name='n256s1/1')
        # n = Conv2d(n, 256, (1, 1), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='out')
        # n = Conv2d(n, , (1, 1), (1, 1), act=tf.identity, W_init=w_init, name='I5')
        # n.outputs = tf.multiply(n.outputs, mask)
        n = SubpixelConv2d(n, scale=4, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        # n = Conv2d(n, 64*times, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        # m = UpSampling2dLayer(m, [2, 2], True, 1, name='upsample15_c')
        # n = P_Conv1(n, m, n_filter=64 * times, name='n256s1/2')
        # n = Conv2d(n, 64, (1, 1), (1, 1), act=tf.identity, W_init=w_init, name='I6')
        # n.outputs = tf.multiply(n.outputs, tf.image.resize_images(mask, size=[mask.shape[1]*2, mask.shape[2]*2], method=0))
        # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')
        # n = InstanceNormLayer(n)
        # n = ConcatLayer([n, UpSampling2dLayer(origin, [4, 4], True, 1, name='Upsample_out')], name='concat3')
        # n1 = UpSampling2dLayer(origin, [4, 4], True, 1, name='Upsample_out')
        # n1 = InstanceNormLayer(n1, name='innorm')
        # n = ConcatLayer([n, n1], name='concat5')
        n = Conv2d(n, 1, (1, 1), (1, 1), act=tf.nn.relu, padding='VALID', W_init=w_init, name='out')

        # n.outputs = tf.multiply(n.outputs, tf.image.resize_images(mask, [mask.shape[1] * 4, mask.shape[2] * 4], method=0))
        # n.outputs = tf.multiply(n.outputs, mask)
        return n


def RDBs_Network(input_image, num_blocks=5, block_layers=3, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    ks, kn = 3, 64
    with tf.variable_scope('RDBS', reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        F_IN = InputLayer(input_image, 'F_IN')
        F_1 = Conv2d(F_IN, kn, W_init=w_init(stddev=np.sqrt(2.0/ks**2)), name='F_1')
        F0 = Conv2d(F_1, kn, W_init=w_init(stddev=np.sqrt(2.0/ks**2/64)), name='F_0')

        rdb_concat = list()
        rdb_in = F0
        for i in range(1, num_blocks+1):
            x = rdb_in
            for j in range(1, block_layers+1):
                temp = Conv2d(x, kn, act=tf.nn.relu, W_init=w_init(stddev=np.sqrt(2.0/ks**2/(kn * j))), name='RDB_C_%d_%d' % (i, j))
                # x = ElementwiseLayer([x, temp], tf.concat, name='RDB_C_%d_%d_concat' % (i, j), axis=-1)
                x = ConcatLayer([x, temp], name='RDB_C_%d_%d_concat' % (i, j))
            x = Conv2d(x, kn, filter_size=(1, 1), W_init=w_init(stddev=np.sqrt(2.0/1/(ks * 2))), name='RDB_C_%d_%d' % (i, block_layers+1))
            x = BatchNormLayer(x, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='RDB_BN_%d_%d' % (i, block_layers+1))
            rdb_in = ElementwiseLayer([x, rdb_in], tf.add, name='RDB_C_%d_add' % i)
            rdb_concat.append(rdb_in)

        FD = ConcatLayer(rdb_concat, name='FD')
        # FD = ElementwiseLayer(rdb_concat, tf.concat, name='FD', axis=-1)

        FGF1 = Conv2d(FD, kn, filter_size=(1, 1), W_init=w_init(stddev=np.sqrt(2.0/1/(kn * num_blocks))), name='FGF1')
        # FGF1 = BatchNormLayer(FGF1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='FGF1_BN')

        FGF2 = Conv2d(FGF1, kn, W_init=w_init(stddev=np.sqrt(2.0/ks**2/kn)), name='FGF2')
        FGF2 = BatchNormLayer(FGF2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='FGF2_BN')

        FDF = ElementwiseLayer([FGF2, F_1], tf.add, name='last_add')

        FUC1 = Conv2d(FDF, kn, filter_size=(5, 5), act=tf.nn.relu, W_init=w_init(stddev=np.sqrt(2.0/25/kn)), name='FUC1')
        FUC2 = Conv2d(FUC1, kn//2, act=tf.nn.relu, W_init=w_init(stddev=np.sqrt(2.0/9/64)), name='FUC2')
        FUC3 = Conv2d(FUC2, kn//4, W_init=w_init(stddev=np.sqrt(2.0/9/32)), name='FUC3')
        FU = SubpixelConv2d(FUC3, scale=4, n_out_channel=None, name='pixelshufflerx')

        IHR = Conv2d(FU, 1, filter_size=(1, 1), act=tf.nn.tanh, W_init=w_init(stddev=0.02), name='OUTPUT')
        return IHR


def SRGAN_g(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        n = Conv2d(n, 1, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n


def SRGAN_d(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, name='h0/c')

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = BatchNormLayer(net_h1, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h1/bn')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h2/bn')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h3/bn')
        net_h4 = Conv2d(net_h3, df_dim*16, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h4 = BatchNormLayer(net_h4, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h4/bn')
        net_h5 = Conv2d(net_h4, df_dim*32, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h5/c')
        net_h5 = BatchNormLayer(net_h5, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h5/bn')
        net_h6 = Conv2d(net_h5, df_dim*16, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
        net_h6 = BatchNormLayer(net_h6, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h6/bn')
        net_h7 = Conv2d(net_h6, df_dim*8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')
        net_h7 = BatchNormLayer(net_h7, is_train=is_train, gamma_init=gamma_init, name='h7/bn')

        net = Conv2d(net_h7, df_dim*2, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn')
        net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn2')
        net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
        net = BatchNormLayer(net, is_train=is_train, gamma_init=gamma_init, name='res/bn3')
        net_h8 = ElementwiseLayer([net_h7, net], combine_fn=tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='ho/dense')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

    return net_ho, logits


def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    import time
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR

        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else: # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool3')
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool4')                               # (batch_size, 14, 14, 512)
        conv = network
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool5')                               # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv