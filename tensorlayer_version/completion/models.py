import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *




b_init = None  # tf.constant_initializer(value=0.0)
g_init = tf.random_normal_initializer(1., 0.02)
w_mask_init = tf.constant_initializer(value=1.)
b_mask_init = tf.constant_initializer(value=0.)
ks, kn = 3, 64


# def P_Conv(net, mask, n_filter=32, filter_size=3, stride=1, name=''):
#
#     img_patch = tf.extract_image_patches(net.outputs, ksizes=[1, filter_size, filter_size, 1],
#                                          strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
#     img_patch = tf.depth_to_space(img_patch, filter_size)
#
#     # mask = tf.nn.relu(tf.sign(tf.abs(net.outputs)-(1e-10)))
#     # mask = InputLayer(mask)
#
#     mask_patch = tf.extract_image_patches(mask.outputs, ksizes=[1, filter_size, filter_size, 1],
#                                           strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
#     count = tf.reduce_sum(mask_patch, axis=-1, keepdims=True)
#     mask_patch = tf.div(mask_patch, count-1e-20)
#     mask_patch = tf.depth_to_space(mask_patch, filter_size)
#
#     img_patch = tf.multiply(img_patch, mask_patch)
#     n = InputLayer(img_patch, name=name+'_input')
#     n = Conv2d(n, n_filter=n_filter, filter_size=(filter_size, filter_size), padding='VALID', W_init=w_init, strides=(filter_size, filter_size), name=name+'_depth')
#     # mask.outputs = tf.nn.relu(tf.sign(n.outputs + 1e10))
#     m = Conv2d(mask, n_filter=n_filter, filter_size=(filter_size, filter_size), W_init=w_mask_init, b_init=b_mask_init,
#                strides=(stride, stride), trainable=False, name=name + '_mask')
#     m.outputs = tf.sign(m.outputs)
#
#     return n, m


def P_Conv(net, n_filter=32, filter_size=3, stride=1, name=''):
    w_init = tf.contrib.layers.xavier_initializer()
    img_patch = tf.extract_image_patches(net.outputs, ksizes=[1, filter_size, filter_size, 1],
                                         strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
    img_patch = tf.depth_to_space(img_patch, filter_size)

    mask = tf.nn.relu(tf.sign(tf.abs(net.outputs)-(1e-20)))

    mask_patch = tf.extract_image_patches(mask, ksizes=[1, filter_size, filter_size, 1],
                                          strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
    count = tf.reduce_sum(mask_patch, axis=-1, keepdims=True)
    mask_patch = tf.div(mask_patch, count-1e-20)
    mask_patch = tf.depth_to_space(mask_patch, filter_size)

    img_patch = tf.multiply(img_patch, mask_patch)
    n = InputLayer(img_patch, name=name+'_input')
    n = Conv2d(n, n_filter=n_filter, filter_size=(filter_size, filter_size), padding='VALID', W_init=w_init, strides=(filter_size, filter_size), name=name+'_depth')

    return n


def Unetwork(input, is_train=False, reuse=False):
    num_filter = 8

    with tf.variable_scope(Unetwork.__name__, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(input, name='depth')

        # Encoder Section
        # PConv 1
        conv1 = P_Conv(n,  n_filter=num_filter, filter_size=7, stride=2, name='pconv1')

        # PConv 2
        conv2 = P_Conv(conv1, filter_size=3, n_filter=num_filter * 5, stride=2, name='pconv2')
        # conv2 = BatchNormLayer(conv2, act=tf.nn.leaky_relu, is_train=istrain, name='bn2')
        conv2 = InstanceNormLayer(conv2, act=tf.nn.elu, name='bn2')

        #  PConv 3
        conv3 = P_Conv(conv2, n_filter=num_filter * 4, filter_size=3, stride=2, name='pconv3')
        # conv3 = BatchNormLayer(conv3, act=tf.nn.leaky_relu, is_train=istrain, name='bn3')
        conv3 = InstanceNormLayer(conv3, act=tf.nn.elu, name='bn3')

        # # PConv 4
        conv4 = P_Conv(conv3, n_filter=num_filter * 8, filter_size=3, stride=2, name='pconv4')
        # conv4 = BatchNormLayer(conv4, act=tf.nn.leaky_relu, is_train=istrain, name='bn4')
        conv4 = InstanceNormLayer(conv4, act=tf.nn.elu, name='bn4')

        # PConv 5
        conv5 = P_Conv(conv4, n_filter=num_filter * 16, filter_size=3, stride=2, name='pconv5')
        conv5 = InstanceNormLayer(conv5, act=tf.nn.elu, name='bn5')
        # conv5 = BatchNormLayer(conv5, act=tf.nn.leaky_relu, is_train=istrain, name='bn5')

        # PConv 6
        conv6 = P_Conv(conv5,  n_filter=num_filter*32, filter_size=3, stride=2, name='pconv6')
        conv6 = InstanceNormLayer(conv6, act=tf.nn.elu, name='bn6')
        # conv6 = BatchNormLayer(conv6, act=tf.nn.leaky_relu, is_train=istrain, name='bn6')

        # PConv 7
        conv7 = P_Conv(conv6, n_filter=num_filter*32, filter_size=3, stride=2, name='pconv7')
        conv7 = InstanceNormLayer(conv7, act=tf.nn.elu, name='bn7')
        # # conv7 = BatchNormLayer(conv7, act=tf.nn.relu, is_train=istrain, name='bn7')

        # # PConv 8
        # # conv8,  mask8 = P_Conv(conv7, mask7, n_filter=num_filter*8, filter_size=3, stride=2, name='pconv8')
        # # conv8 = BatchNormLayer(conv8, act=tf.nn.relu, is_train=istrain, name='bn8')

        # # Decoder Section
        # # # DPConv 1
        # # size = conv7.outputs.get_shape().as_list()
        # # upsample8_c = UpSampling2dLayer(conv8, [size[1], size[2]], False, 1, name='upsample8_c')
        # # concat7_u8_c = ConcatLayer([upsample8_c, conv7], name='concat7_u8_c')
        # # conv9, mask9 = P_Conv(concat7_u8_c, concat7_u8_m, n_filter=num_filter*8, name='pconv9')
        # # conv9 = BatchNormLayer(conv9, act=tf.nn.leaky_relu, is_train=istrain, name='bn9')

        # DPConv 2
        size = conv6.outputs.get_shape().as_list()
        upsample9_c = UpSampling2dLayer(conv7, [size[1], size[2]], False, 1, name='upsample9_c')
        concat6_u9_c = ConcatLayer([upsample9_c, conv6], name='concat6_u9_c')
        conv10 = P_Conv(concat6_u9_c, n_filter=num_filter*8, name='pconv10')
        conv10 = InstanceNormLayer(conv10, act=tf.nn.elu, name='bn10')
        # conv10 = BatchNormLayer(conv10, act=tf.nn.leaky_relu, is_train=istrain, name='bn10')

        # DPConv 3
        size = conv5.outputs.get_shape().as_list()
        upsample10_c = UpSampling2dLayer(conv10, [size[1], size[2]], False, 1, name='upsample10_c')
        # upsample10_c = UpSampling2dLayer(conv6, [size[1], size[2]], False, 1, name='upsample10_c')
        concat5_u10_c = ConcatLayer([upsample10_c, conv5], name='concat5_u10_c')
        conv11 = P_Conv(concat5_u10_c, n_filter=num_filter*8, name='pconv11')
        conv11 = InstanceNormLayer(conv11, act=tf.nn.elu, name='bn11')
        # conv11 = BatchNormLayer(conv11, act=tf.nn.leaky_relu, is_train=istrain, name='bn11')

        # DPConv 4
        size = conv4.outputs.get_shape().as_list()
        upsample11_c = UpSampling2dLayer(conv11, [size[1], size[2]], False, 1, name='upsample11_c')
        # upsample11_c = UpSampling2dLayer(conv5, [size[1], size[2]], False, 1, name='upsample11_c')
        concat4_u11_c = ConcatLayer([upsample11_c, conv4], name='concat4_u11_c')
        conv12 = P_Conv(concat4_u11_c, n_filter=num_filter*8, name='pconv12')
        conv12 = InstanceNormLayer(conv12, act=tf.nn.elu, name='bn12')
        # conv12 = BatchNormLayer(conv12, act=tf.nn.leaky_relu, is_train=istrain, name='bn12')

        # DPConv 5
        size = conv3.outputs.get_shape().as_list()
        upsample12_c = UpSampling2dLayer(conv12, [size[1], size[2]], False, 1, name='upsample12_c')
        # upsample12_c = UpSampling2dLayer(conv4, [size[1], size[2]], False, 1, name='upsample12_c')
        concat3_u12_c = ConcatLayer([upsample12_c, conv3], name='concat3_u12_c')
        conv13 = P_Conv(concat3_u12_c, n_filter=num_filter*4, name='pconv13')
        # conv13 = BatchNormLayer(conv13, act=tf.nn.leaky_relu, is_train=istrain, name='bn13')
        conv13 = InstanceNormLayer(conv13, act=tf.nn.elu, name='bn13')

        # DPConv 6
        size = conv2.outputs.get_shape().as_list()
        upsample13_c = UpSampling2dLayer(conv13, [size[1], size[2]], False, 1, name='upsample13_c')
        # upsample13_c = UpSampling2dLayer(conv3, [size[1], size[2]], False, 1, name='upsample13_c')
        concat2_u13_c = ConcatLayer([upsample13_c, conv2], name='concat2_u13_c')
        conv14 = P_Conv(concat2_u13_c, n_filter=num_filter*2, name='pconv14')
        # conv14 = BatchNormLayer(conv14, act=tf.nn.leaky_relu, is_train=istrain, name='bn14')
        conv14 = InstanceNormLayer(conv14, act=tf.nn.elu, name='bn14')

        # DPConv 7
        size = conv1.outputs.get_shape().as_list()
        upsample14_c = UpSampling2dLayer(conv14, [size[1], size[2]], False, 1, name='upsample14_c')
        # upsample14_c = UpSampling2dLayer(conv2, [size[1], size[2]], False, 1, name='upsample14_c')
        concat1_u14_c = ConcatLayer([upsample14_c, conv1], name='concat1_u14_c')
        conv15 = P_Conv(concat1_u14_c, n_filter=num_filter, name='pconv15')
        # conv15 = BatchNormLayer(conv15, act=tf.nn.leaky_relu, is_train=istrain, name='bn15')
        conv15 = InstanceNormLayer(conv15, act=tf.nn.elu, name='bn15')

        # DPConv 8
        size = n.outputs.get_shape().as_list()
        upsample15_c = UpSampling2dLayer(conv15, [size[1], size[2]], False, 1, name='upsample15_c')
        # upsample15_c = UpSampling2dLayer(conv1, [size[1], size[2]], False, 1, name='upsample15_c')
        concat0_u15_c = ConcatLayer([upsample15_c, n], name='concat0_u15_c')
        conv16 = P_Conv(concat0_u15_c, n_filter=1, name='pconv16')

        return conv16


def Unetwork_NMASK(input, is_train=False, reuse=False):
    num_filter = 8

    with tf.variable_scope(Unetwork_NMASK.__name__, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(input, name='depth')

        # Encoder Section
        # PConv 1
        # conv1 = P_Conv(n,  n_filter=num_filter, filter_size=7, stride=2, name='pconv1')
        conv1 = Conv2d(n, n_filter=num_filter, filter_size=7, strides=(2, 2), name='pconv1')

        # PConv 2
        # conv2 = P_Conv(conv1, filter_size=3, n_filter=num_filter * 5, stride=2, name='pconv2')
        conv2 = Conv2d(conv1, filter_size=3, n_filter=num_filter * 5, strides=(2, 2), name='pconv2')
        # conv2 = BatchNormLayer(conv2, act=tf.nn.leaky_relu, is_train=istrain, name='bn2')
        conv2 = InstanceNormLayer(conv2, act=tf.nn.elu, name='bn2')

        #  PConv 3
        conv3 = Conv2d(conv2, n_filter=num_filter * 4, filter_size=3, strides=(2, 2), name='pconv3')
        # conv3 = P_Conv(conv2, n_filter=num_filter * 4, filter_size=3, stride=2, name='pconv3')
        # conv3 = BatchNormLayer(conv3, act=tf.nn.leaky_relu, is_train=istrain, name='bn3')
        conv3 = InstanceNormLayer(conv3, act=tf.nn.elu, name='bn3')

        # # PConv 4
        conv4 = Conv2d(conv3, n_filter=num_filter * 8, filter_size=3, strides=(2, 2), name='pconv4')
        # conv4 = P_Conv(conv3, n_filter=num_filter * 8, filter_size=3, stride=2, name='pconv4')
        # conv4 = BatchNormLayer(conv4, act=tf.nn.leaky_relu, is_train=istrain, name='bn4')
        conv4 = InstanceNormLayer(conv4, act=tf.nn.elu, name='bn4')

        # PConv 5
        conv5 = Conv2d(conv4, n_filter=num_filter * 16, filter_size=3, strides=(2, 2), name='pconv5')
        # conv5 = P_Conv(conv4, n_filter=num_filter * 16, filter_size=3, stride=2, name='pconv5')
        conv5 = InstanceNormLayer(conv5, act=tf.nn.elu, name='bn5')
        # conv5 = BatchNormLayer(conv5, act=tf.nn.leaky_relu, is_train=istrain, name='bn5')

        # PConv 6
        # conv6 = P_Conv(conv5,  n_filter=num_filter*32, filter_size=3, stride=2, name='pconv6')
        # conv6 = InstanceNormLayer(conv6, act=tf.nn.elu, name='bn6')
        # conv6 = BatchNormLayer(conv6, act=tf.nn.leaky_relu, is_train=istrain, name='bn6')

        # PConv 7
        # conv7 = P_Conv(conv6, n_filter=num_filter*32, filter_size=3, stride=2, name='pconv7')
        # conv7 = InstanceNormLayer(conv7, act=tf.nn.elu, name='bn7')
        # # conv7 = BatchNormLayer(conv7, act=tf.nn.relu, is_train=istrain, name='bn7')

        # # PConv 8
        # # conv8,  mask8 = P_Conv(conv7, mask7, n_filter=num_filter*8, filter_size=3, stride=2, name='pconv8')
        # # conv8 = BatchNormLayer(conv8, act=tf.nn.relu, is_train=istrain, name='bn8')

        # # Decoder Section
        # # # DPConv 1
        # # size = conv7.outputs.get_shape().as_list()
        # # upsample8_c = UpSampling2dLayer(conv8, [size[1], size[2]], False, 1, name='upsample8_c')
        # # concat7_u8_c = ConcatLayer([upsample8_c, conv7], name='concat7_u8_c')
        # # conv9, mask9 = P_Conv(concat7_u8_c, concat7_u8_m, n_filter=num_filter*8, name='pconv9')
        # # conv9 = BatchNormLayer(conv9, act=tf.nn.leaky_relu, is_train=istrain, name='bn9')

        # DPConv 2
        # size = conv6.outputs.get_shape().as_list()
        # upsample9_c = UpSampling2dLayer(conv7, [size[1], size[2]], False, 1, name='upsample9_c')
        # concat6_u9_c = ConcatLayer([upsample9_c, conv6], name='concat6_u9_c')
        # conv10 = P_Conv(concat6_u9_c, n_filter=num_filter*8, name='pconv10')
        # conv10 = InstanceNormLayer(conv10, act=tf.nn.elu, name='bn10')
        # conv10 = BatchNormLayer(conv10, act=tf.nn.leaky_relu, is_train=istrain, name='bn10')

        # DPConv 3
        # size = conv5.outputs.get_shape().as_list()
        # upsample10_c = UpSampling2dLayer(conv10, [size[1], size[2]], False, 1, name='upsample10_c')
        # # upsample10_c = UpSampling2dLayer(conv6, [size[1], size[2]], False, 1, name='upsample10_c')
        # concat5_u10_c = ConcatLayer([upsample10_c, conv5], name='concat5_u10_c')
        # conv11 = P_Conv(concat5_u10_c, n_filter=num_filter*8, name='pconv11')
        # conv11 = InstanceNormLayer(conv11, act=tf.nn.elu, name='bn11')
        # conv11 = BatchNormLayer(conv11, act=tf.nn.leaky_relu, is_train=istrain, name='bn11')

        # DPConv 4
        size = conv4.outputs.get_shape().as_list()
        upsample11_c = UpSampling2dLayer(conv5, [size[1], size[2]], False, 1, name='upsample11_c')
        # upsample11_c = UpSampling2dLayer(conv5, [size[1], size[2]], False, 1, name='upsample11_c')
        concat4_u11_c = ConcatLayer([upsample11_c, conv4], name='concat4_u11_c')
        conv12 = Conv2d(concat4_u11_c, n_filter=num_filter*8, name='pconv12')
        # conv12 = P_Conv(concat4_u11_c, n_filter=num_filter * 8, name='pconv12')
        # conv12 = P_Conv(concat4_u11_c, n_filter=num_filter * 8, name='pconv12')
        conv12 = InstanceNormLayer(conv12, act=tf.nn.elu, name='bn12')
        # conv12 = BatchNormLayer(conv12, act=tf.nn.leaky_relu, is_train=istrain, name='bn12')

        # DPConv 5
        size = conv3.outputs.get_shape().as_list()
        upsample12_c = UpSampling2dLayer(conv12, [size[1], size[2]], False, 1, name='upsample12_c')
        # upsample12_c = UpSampling2dLayer(conv4, [size[1], size[2]], False, 1, name='upsample12_c')
        concat3_u12_c = ConcatLayer([upsample12_c, conv3], name='concat3_u12_c')
        conv13 = Conv2d(concat3_u12_c, n_filter=num_filter*4, name='pconv13')
        # conv13 = P_Conv(concat3_u12_c, n_filter=num_filter * 4, name='pconv13')
        # conv13 = BatchNormLayer(conv13, act=tf.nn.leaky_relu, is_train=istrain, name='bn13')
        conv13 = InstanceNormLayer(conv13, act=tf.nn.elu, name='bn13')

        # DPConv 6
        size = conv2.outputs.get_shape().as_list()
        upsample13_c = UpSampling2dLayer(conv13, [size[1], size[2]], False, 1, name='upsample13_c')
        # upsample13_c = UpSampling2dLayer(conv3, [size[1], size[2]], False, 1, name='upsample13_c')
        concat2_u13_c = ConcatLayer([upsample13_c, conv2], name='concat2_u13_c')
        conv14 = Conv2d(concat2_u13_c, n_filter=num_filter*2, name='pconv14')
        # conv14 = P_Conv(concat2_u13_c, n_filter=num_filter * 2, name='pconv14')
        # conv14 = BatchNormLayer(conv14, act=tf.nn.leaky_relu, is_train=istrain, name='bn14')
        conv14 = InstanceNormLayer(conv14, act=tf.nn.elu, name='bn14')

        # DPConv 7
        size = conv1.outputs.get_shape().as_list()
        upsample14_c = UpSampling2dLayer(conv14, [size[1], size[2]], False, 1, name='upsample14_c')
        # upsample14_c = UpSampling2dLayer(conv2, [size[1], size[2]], False, 1, name='upsample14_c')
        concat1_u14_c = ConcatLayer([upsample14_c, conv1], name='concat1_u14_c')
        # conv15 = P_Conv(concat1_u14_c, n_filter=num_filter, name='pconv15')
        conv15 = Conv2d(concat1_u14_c, n_filter=num_filter, name='pconv15')
        # conv15 = BatchNormLayer(conv15, act=tf.nn.leaky_relu, is_train=istrain, name='bn15')
        conv15 = InstanceNormLayer(conv15, act=tf.nn.elu, name='bn15')

        # DPConv 8
        size = n.outputs.get_shape().as_list()
        upsample15_c = UpSampling2dLayer(conv15, [size[1], size[2]], False, 1, name='upsample15_c')
        # upsample15_c = UpSampling2dLayer(conv1, [size[1], size[2]], False, 1, name='upsample15_c')
        concat0_u15_c = ConcatLayer([upsample15_c, n], name='concat0_u15_c')
        # conv16 = P_Conv(concat0_u15_c, n_filter=1, name='pconv16')
        conv16 = Conv2d(concat0_u15_c, n_filter=1, name='pconv16')

        return conv16


# def refine_net(depth, gray):
#     filter_size, stride = 21, 1
#     gray_patch = tf.extract_image_patches(gray, ksizes=[1, filter_size, filter_size, 1],
#                                           strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
#     depth_patch = tf.extract_image_patches(depth, ksizes=[1, filter_size, filter_size, 1],
#                                            strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
#     # center_depth = depth_patch[:, :, :, filter_size ** 2 // 2:filter_size ** 2 // 2 + 1]
#     # mask_depth = tf.sign(tf.abs(depth_patch-center_depth))
#     # depth_patch = tf.multiply(depth_patch, mask_depth)
#
#     center = gray_patch[:, :, :, filter_size ** 2 // 2:filter_size ** 2 // 2 + 1]
#
#     w_similar = tf.exp(-tf.div(tf.square(gray_patch-center), 2*6**2))
#     w_similar = tf.div(w_similar, tf.reduce_sum(w_similar, axis=-1, keepdims=True))
#
#     gray_patch = tf.depth_to_space(w_similar, filter_size)
#     depth_patch = tf.depth_to_space(depth_patch, filter_size)
#
#     depth_patch = tf.multiply(depth_patch, gray_patch)
#     name = 'jbf'
#     n = InputLayer(depth_patch, name=name + '_input')
#     w_mask_init = tf.constant_initializer(value=1.)
#     b_mask_init = tf.constant_initializer(value=0.)
#     n = Conv2d(n, n_filter=1, filter_size=(filter_size, filter_size), padding='VALID', W_init=w_mask_init,
#                b_init=b_mask_init,
#                strides=(filter_size, filter_size), name=name + '_depth')
#     return n

def P_Conv_NBN(net, n_filter=32, filter_size=3, stride=1, name=''):
    w_init = tf.random_normal_initializer(mean=0., stddev=1.0)
    img_patch = tf.extract_image_patches(net.outputs, ksizes=[1, filter_size, filter_size, 1],
                                         strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
    img_patch = tf.depth_to_space(img_patch, filter_size)

    mask = tf.nn.relu(tf.sign(tf.abs(net.outputs)-(1e-20)))

    mask_patch = tf.extract_image_patches(mask, ksizes=[1, filter_size, filter_size, 1],
                                          strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
    count = tf.reduce_sum(mask_patch, axis=-1, keepdims=True)
    mask_patch = tf.div(mask_patch, count-1e-20)
    mask_patch = tf.depth_to_space(mask_patch, filter_size)

    img_patch = tf.multiply(img_patch, mask_patch)
    n = InputLayer(img_patch, name=name+'_input')
    n = Conv2d(n, n_filter=n_filter, filter_size=(filter_size, filter_size), padding='VALID', W_init=w_init, strides=(filter_size, filter_size), name=name+'_depth')

    return n


def Unetwork_NBN(input, is_train=False, reuse=False):
    num_filter = 8
    with tf.variable_scope(Unetwork_NBN.__name__, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        mask = tf.sign(input)
        # m = InputLayer(mask, name='mask')
        # input = input - tf.abs(mask-1)*1e20
        n = InputLayer(input, name='depth')


        # Encoder Section
        # PConv 1
        conv1 = P_Conv_NBN(n,  n_filter=num_filter, filter_size=3, stride=2, name='pconv1')
        # all_paras.append(conv1.all_params)
        # conv1, mask1 = P_Conv(n, m, n_filter=num_filter, filter_size=3, stride=1, name='pconv1_1')
        # conv1, mask1 = P_Conv(conv1, mask1, n_filter=num_filter, filter_size=3, stride=1, name='pconv1_2')
        # conv1, mask1 = P_Conv(conv1, mask1, n_filter=1, filter_size=3, stride=1, name='pconv1_3')

        # PConv 2
        conv2 = P_Conv_NBN(conv1, filter_size=3, n_filter=num_filter, stride=1, name='pconv2_0')
        conv2 = P_Conv_NBN(conv2, filter_size=3, n_filter=num_filter * 2, stride=2, name='pconv2')
        # conv2 = BatchNormLayer(conv2, act=tf.nn.leaky_relu, is_train=istrain, name='bn2')
        # conv2 = InstanceNormLayer(conv2, act=tf.nn.elu, name='bn2')

        #  PConv 3
        conv3 = P_Conv_NBN(conv2, n_filter=num_filter * 2, filter_size=3, stride=1, name='pconv3_0')
        conv3 = P_Conv_NBN(conv3, n_filter=num_filter * 4, filter_size=3, stride=2, name='pconv3')
        # conv3 = BatchNormLayer(conv3, act=tf.nn.leaky_relu, is_train=istrain, name='bn3')
        # conv3 = InstanceNormLayer(conv3, act=tf.nn.elu, name='bn3')

        # # PConv 4
        conv4 = P_Conv_NBN(conv3, n_filter=num_filter * 4, filter_size=3, stride=1, name='pconv4')
        # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 16, filter_size=3, stride=2, name='pconv4_1')
        # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 16, filter_size=3, stride=1, name='pconv4_2')
        conv4 = P_Conv_NBN(conv4, n_filter=num_filter * 8, filter_size=3, stride=2, name='pconv4_3')
        # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_4')
        # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_5')
        # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_6')
        # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_7')
        # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_8')
        # conv4 = BatchNormLayer(conv4, act=tf.nn.leaky_relu, is_train=istrain, name='bn4')
        # conv4 = InstanceNormLayer(conv4, act=tf.nn.elu, name='bn4')

        # PConv 5
        conv5 = P_Conv_NBN(conv4, n_filter=num_filter * 8, filter_size=3, stride=1, name='pconv5')
        conv5 = P_Conv_NBN(conv5, n_filter=num_filter * 16, filter_size=3, stride=2, name='pconv5_1')
        conv5 = P_Conv_NBN(conv5, n_filter=num_filter * 16, filter_size=3, stride=1, name='pconv5_2')
        conv5 = P_Conv_NBN(conv5, n_filter=num_filter * 16, filter_size=3, stride=1, name='pconv5_3')
        # conv5 = BatchNormLayer(conv5, act=tf.nn.leaky_relu, is_train=istrain, name='bn5')

        # # PConv 6
        # conv6,  mask6 = P_Conv(conv5, mask5, n_filter=num_filter*8, filter_size=5, stride=2, name='pconv6')
        # conv6 = BatchNormLayer(conv6, act=tf.nn.leaky_relu, is_train=istrain, name='bn6')

        # # PConv 7
        # # conv7, mask7 = P_Conv(conv6, mask6, n_filter=num_filter*8, filter_size=3, stride=2, name='pconv7')
        # # conv7 = BatchNormLayer(conv7, act=tf.nn.relu, is_train=istrain, name='bn7')

        # # # PConv 8
        # # conv8,  mask8 = P_Conv(conv7, mask7, n_filter=num_filter*8, filter_size=3, stride=2, name='pconv8')
        # # conv8 = BatchNormLayer(conv8, act=tf.nn.relu, is_train=istrain, name='bn8')

        # # Decoder Section
        # # # DPConv 1
        # # size = conv7.outputs.get_shape().as_list()
        # # upsample8_c = UpSampling2dLayer(conv8, [size[1], size[2]], False, 1, name='upsample8_c')
        # # upsample8_m = UpSampling2dLayer(mask8, [size[1], size[2]], False, 1, name='upsample8_m')
        # # concat7_u8_c = ConcatLayer([upsample8_c, conv7], name='concat7_u8_c')
        # # concat7_u8_m = ConcatLayer([upsample8_m, mask7], name='concat7_u8_m')
        # # conv9, mask9 = P_Conv(concat7_u8_c, concat7_u8_m, n_filter=num_filter*8, name='pconv9')
        # # conv9 = BatchNormLayer(conv9, act=tf.nn.leaky_relu, is_train=istrain, name='bn9')

        # # # DPConv 2
        # size = conv6.outputs.get_shape().as_list()
        # upsample9_c = UpSampling2dLayer(conv7, [size[1], size[2]], False, 1, name='upsample9_c')
        # upsample9_m = UpSampling2dLayer(mask7, [size[1], size[2]], False, 1, name='upsample9_m')
        # concat6_u9_c = ConcatLayer([upsample9_c, conv6], name='concat6_u9_c')
        # concat6_u9_m = ConcatLayer([upsample9_m, mask6], name='concat6_u9_m')
        # conv10, mask10 = P_Conv(concat6_u9_c, concat6_u9_m, n_filter=num_filter*8, name='pconv10')
        # # conv10 = BatchNormLayer(conv10, act=tf.nn.leaky_relu, is_train=istrain, name='bn10')

        # # DPConv 3
        # size = conv5.outputs.get_shape().as_list()
        # # upsample10_c = UpSampling2dLayer(conv10, [size[1], size[2]], False, 1, name='upsample10_c')
        # # upsample10_m = UpSampling2dLayer(mask10, [size[1], size[2]], False, 1, name='upsample10_m')
        # upsample10_c = UpSampling2dLayer(conv6, [size[1], size[2]], False, 1, name='upsample10_c')
        # upsample10_m = UpSampling2dLayer(mask6, [size[1], size[2]], False, 1, name='upsample10_m')
        # concat5_u10_c = ConcatLayer([upsample10_c, conv5], name='concat5_u10_c')
        # concat5_u10_m = ConcatLayer([upsample10_m, mask5], name='concat5_u10_m')
        # conv11, mask11 = P_Conv(concat5_u10_c, concat5_u10_m, n_filter=num_filter*8, name='pconv11')
        # conv11 = BatchNormLayer(conv11, act=tf.nn.leaky_relu, is_train=istrain, name='bn11')

        # DPConv 4
        size = conv4.outputs.get_shape().as_list()
        # upsample11_c = UpSampling2dLayer(conv11, [size[1], size[2]], False, 1, name='upsample11_c')
        # upsample11_m = UpSampling2dLayer(mask11, [size[1], size[2]], False, 1, name='upsample11_m')
        upsample11_c = UpSampling2dLayer(conv5, [size[1], size[2]], False, 1, name='upsample11_c')
        # upsample11_m = UpSampling2dLayer(mask5, [size[1], size[2]], False, 1, name='upsample11_m')
        concat4_u11_c = ConcatLayer([upsample11_c, conv4], name='concat4_u11_c')
        # concat4_u11_m = ConcatLayer([upsample11_m, mask4], name='concat4_u11_m')
        conv12 = P_Conv_NBN(concat4_u11_c, n_filter=num_filter*8, name='pconv12')
        # conv12 = BatchNormLayer(conv12, act=tf.nn.leaky_relu, is_train=istrain, name='bn12')

        # # DPConv 5
        size = conv3.outputs.get_shape().as_list()
        upsample12_c = UpSampling2dLayer(conv12, [size[1], size[2]], False, 1, name='upsample12_c')
        # upsample12_m = UpSampling2dLayer(mask12, [size[1], size[2]], False, 1, name='upsample12_m')
        # upsample12_c = UpSampling2dLayer(conv4, [size[1], size[2]], False, 1, name='upsample12_c')
        # upsample12_m = UpSampling2dLayer(mask4, [size[1], size[2]], False, 1, name='upsample12_m')
        concat3_u12_c = ConcatLayer([upsample12_c, conv3], name='concat3_u12_c')
        # concat3_u12_m = ConcatLayer([upsample12_m, mask3], name='concat3_u12_m')
        conv13 = P_Conv_NBN(concat3_u12_c, n_filter=num_filter*4, name='pconv13')
        # conv13 = BatchNormLayer(conv13, act=tf.nn.leaky_relu, is_train=istrain, name='bn13')
        # conv13 = InstanceNormLayer(conv13, act=tf.nn.elu, name='bn13')

        # DPConv 6
        size = conv2.outputs.get_shape().as_list()
        upsample13_c = UpSampling2dLayer(conv13, [size[1], size[2]], False, 1, name='upsample13_c')
        # upsample13_m = UpSampling2dLayer(mask13, [size[1], size[2]], False, 1, name='upsample13_m')
        # upsample13_c = UpSampling2dLayer(conv3, [size[1], size[2]], False, 1, name='upsample13_c')
        # upsample13_m = UpSampling2dLayer(mask3, [size[1], size[2]], False, 1, name='upsample13_m')
        concat2_u13_c = ConcatLayer([upsample13_c, conv2], name='concat2_u13_c')
        # concat2_u13_m = ConcatLayer([upsample13_m, mask2], name='concat2_u13_m')
        conv14 = P_Conv_NBN(concat2_u13_c, n_filter=num_filter*2, name='pconv14')
        # conv14 = BatchNormLayer(conv14, act=tf.nn.leaky_relu, is_train=istrain, name='bn14')
        # conv14 = InstanceNormLayer(conv14, act=tf.nn.elu, name='bn14')

        # DPConv 7
        size = conv1.outputs.get_shape().as_list()
        upsample14_c = UpSampling2dLayer(conv14, [size[1], size[2]], False, 1, name='upsample14_c')
        # upsample14_m = UpSampling2dLayer(mask14, [size[1], size[2]], False, 1, name='upsample14_m')
        # upsample14_c = UpSampling2dLayer(conv2, [size[1], size[2]], False, 1, name='upsample14_c')
        # upsample14_m = UpSampling2dLayer(mask2, [size[1], size[2]], False, 1, name='upsample14_m')
        concat1_u14_c = ConcatLayer([upsample14_c, conv1], name='concat1_u14_c')
        # concat1_u14_m = ConcatLayer([upsample14_m, mask1], name='concat1_u14_m')
        conv15 = P_Conv_NBN(concat1_u14_c, n_filter=num_filter, name='pconv15')
        # conv15 = BatchNormLayer(conv15, act=tf.nn.leaky_relu, is_train=istrain, name='bn15')
        # conv15 = InstanceNormLayer(conv15, act=tf.nn.elu, name='bn15')

        # DPConv 8
        size = n.outputs.get_shape().as_list()
        upsample15_c = UpSampling2dLayer(conv15, [size[1], size[2]], False, 1, name='upsample15_c')
        # upsample15_m = UpSampling2dLayer(mask15, [size[1], size[2]], False, 1, name='upsample15_m')
        # upsample15_c = UpSampling2dLayer(conv1, [size[1], size[2]], False, 1, name='upsample15_c')
        # upsample15_m = UpSampling2dLayer(mask1, [size[1], size[2]], False, 1, name='upsample15_m')
        concat0_u15_c = ConcatLayer([upsample15_c, n], name='concat0_u15_c')
        # concat0_u15_m = ConcatLayer([upsample15_m, m], name='concat0_u15_m')
        conv16 = P_Conv_NBN(concat0_u15_c, n_filter=1, name='pconv16')
        conv16 = Conv2d(conv16, n_filter=1, name='pconv16')

        return conv16


def SRGAN_d(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(SRGAN_d.__name__, reuse=reuse):
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

# def P_Conv(net, mask, n_filter=32, filter_size=3, stride=1, name=''):
#
#     img_patch = tf.extract_image_patches(net.outputs, ksizes=[1, filter_size, filter_size, 1],
#                                          strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
#     img_patch = tf.depth_to_space(img_patch, filter_size)
#
#     mask_patch = tf.extract_image_patches(mask.outputs, ksizes=[1, filter_size, filter_size, 1],
#                                           strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
#     count = tf.reduce_sum(mask_patch, axis=-1, keepdims=True)
#     mask_patch = tf.div(mask_patch, count-1e-20)
#     mask_patch = tf.depth_to_space(mask_patch, filter_size)
#
#     img_patch = tf.multiply(img_patch, mask_patch)
#     n = InputLayer(img_patch, name=name+'_input')
#     n = Conv2d(n, n_filter=n_filter, filter_size=(filter_size, filter_size), padding='VALID', W_init=w_init, strides=(filter_size, filter_size), name=name+'_depth')
#     m = Conv2d(mask, n_filter=n_filter, filter_size=(filter_size, filter_size), W_init=w_mask_init, b_init=b_mask_init,
#                strides=(stride, stride), trainable=False, name=name + '_mask')
#     m.outputs = tf.sign(m.outputs)
#
#     return n, m
#
#
# def Unetwork(input, num_filter=8, istrain=False, reuse=False):
#
#     with tf.variable_scope('UNet', reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         n = InputLayer(input, name='depth')
#         mask = tf.sign(input)
#         m = InputLayer(mask, name='mask')
#
#         # Encoder Section
#         # PConv 1
#         conv1, mask1 = P_Conv(n, m, n_filter=num_filter, filter_size=3, stride=2, name='pconv1')
#         # all_paras.append(conv1.all_params)
#         # conv1, mask1 = P_Conv(n, m, n_filter=num_filter, filter_size=3, stride=1, name='pconv1_1')
#         # conv1, mask1 = P_Conv(conv1, mask1, n_filter=num_filter, filter_size=3, stride=1, name='pconv1_2')
#         # conv1, mask1 = P_Conv(conv1, mask1, n_filter=1, filter_size=3, stride=1, name='pconv1_3')
#
#         # PConv 2
#         conv2, mask2 = P_Conv(conv1, mask1, filter_size=3, n_filter=num_filter, stride=1, name='pconv2_0')
#         conv2,  mask2 = P_Conv(conv2, mask2, filter_size=3, n_filter=num_filter * 2, stride=2, name='pconv2')
#         # conv2 = BatchNormLayer(conv2, act=tf.nn.leaky_relu, is_train=istrain, name='bn2')
#         # conv2 = InstanceNormLayer(conv2, act=tf.nn.elu, name='bn2')
#
#         #  PConv 3
#         conv3, mask3 = P_Conv(conv2, mask2, n_filter=num_filter * 2, filter_size=3, stride=1, name='pconv3_0')
#         conv3, mask3 = P_Conv(conv3, mask3, n_filter=num_filter * 4, filter_size=3, stride=2, name='pconv3')
#         # conv3 = BatchNormLayer(conv3, act=tf.nn.leaky_relu, is_train=istrain, name='bn3')
#         # conv3 = InstanceNormLayer(conv3, act=tf.nn.elu, name='bn3')
#
#         # # PConv 4
#         conv4,  mask4 = P_Conv(conv3, mask3, n_filter=num_filter * 4, filter_size=3, stride=1, name='pconv4')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 16, filter_size=3, stride=2, name='pconv4_1')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 16, filter_size=3, stride=1, name='pconv4_2')
#         conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=3, stride=1, name='pconv4_3')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_4')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_5')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_6')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_7')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_8')
#         # conv4 = BatchNormLayer(conv4, act=tf.nn.leaky_relu, is_train=istrain, name='bn4')
#         # conv4 = InstanceNormLayer(conv4, act=tf.nn.elu, name='bn4')
#
#         # PConv 5
#         conv5, mask5 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=3, stride=1, name='pconv5')
#         conv5, mask5 = P_Conv(conv5, mask5, n_filter=num_filter * 16, filter_size=3, stride=2, name='pconv5_1')
#         conv5, mask5 = P_Conv(conv5, mask5, n_filter=num_filter * 16, filter_size=3, stride=1, name='pconv5_2')
#         conv5, mask5 = P_Conv(conv5, mask5, n_filter=num_filter * 16, filter_size=3, stride=1, name='pconv5_3')
#         # conv5 = BatchNormLayer(conv5, act=tf.nn.leaky_relu, is_train=istrain, name='bn5')
#
#         # # PConv 6
#         # conv6,  mask6 = P_Conv(conv5, mask5, n_filter=num_filter*8, filter_size=5, stride=2, name='pconv6')
#         # conv6 = BatchNormLayer(conv6, act=tf.nn.leaky_relu, is_train=istrain, name='bn6')
#
#         # # PConv 7
#         # # conv7, mask7 = P_Conv(conv6, mask6, n_filter=num_filter*8, filter_size=3, stride=2, name='pconv7')
#         # # conv7 = BatchNormLayer(conv7, act=tf.nn.relu, is_train=istrain, name='bn7')
#
#         # # # PConv 8
#         # # conv8,  mask8 = P_Conv(conv7, mask7, n_filter=num_filter*8, filter_size=3, stride=2, name='pconv8')
#         # # conv8 = BatchNormLayer(conv8, act=tf.nn.relu, is_train=istrain, name='bn8')
#
#         # # Decoder Section
#         # # # DPConv 1
#         # # size = conv7.outputs.get_shape().as_list()
#         # # upsample8_c = UpSampling2dLayer(conv8, [size[1], size[2]], False, 1, name='upsample8_c')
#         # # upsample8_m = UpSampling2dLayer(mask8, [size[1], size[2]], False, 1, name='upsample8_m')
#         # # concat7_u8_c = ConcatLayer([upsample8_c, conv7], name='concat7_u8_c')
#         # # concat7_u8_m = ConcatLayer([upsample8_m, mask7], name='concat7_u8_m')
#         # # conv9, mask9 = P_Conv(concat7_u8_c, concat7_u8_m, n_filter=num_filter*8, name='pconv9')
#         # # conv9 = BatchNormLayer(conv9, act=tf.nn.leaky_relu, is_train=istrain, name='bn9')
#
#         # # # DPConv 2
#         # size = conv6.outputs.get_shape().as_list()
#         # upsample9_c = UpSampling2dLayer(conv7, [size[1], size[2]], False, 1, name='upsample9_c')
#         # upsample9_m = UpSampling2dLayer(mask7, [size[1], size[2]], False, 1, name='upsample9_m')
#         # concat6_u9_c = ConcatLayer([upsample9_c, conv6], name='concat6_u9_c')
#         # concat6_u9_m = ConcatLayer([upsample9_m, mask6], name='concat6_u9_m')
#         # conv10, mask10 = P_Conv(concat6_u9_c, concat6_u9_m, n_filter=num_filter*8, name='pconv10')
#         # # conv10 = BatchNormLayer(conv10, act=tf.nn.leaky_relu, is_train=istrain, name='bn10')
#
#         # # DPConv 3
#         # size = conv5.outputs.get_shape().as_list()
#         # # upsample10_c = UpSampling2dLayer(conv10, [size[1], size[2]], False, 1, name='upsample10_c')
#         # # upsample10_m = UpSampling2dLayer(mask10, [size[1], size[2]], False, 1, name='upsample10_m')
#         # upsample10_c = UpSampling2dLayer(conv6, [size[1], size[2]], False, 1, name='upsample10_c')
#         # upsample10_m = UpSampling2dLayer(mask6, [size[1], size[2]], False, 1, name='upsample10_m')
#         # concat5_u10_c = ConcatLayer([upsample10_c, conv5], name='concat5_u10_c')
#         # concat5_u10_m = ConcatLayer([upsample10_m, mask5], name='concat5_u10_m')
#         # conv11, mask11 = P_Conv(concat5_u10_c, concat5_u10_m, n_filter=num_filter*8, name='pconv11')
#         # conv11 = BatchNormLayer(conv11, act=tf.nn.leaky_relu, is_train=istrain, name='bn11')
#
#         # DPConv 4
#         size = conv4.outputs.get_shape().as_list()
#         # upsample11_c = UpSampling2dLayer(conv11, [size[1], size[2]], False, 1, name='upsample11_c')
#         # upsample11_m = UpSampling2dLayer(mask11, [size[1], size[2]], False, 1, name='upsample11_m')
#         upsample11_c = UpSampling2dLayer(conv5, [size[1], size[2]], False, 1, name='upsample11_c')
#         upsample11_m = UpSampling2dLayer(mask5, [size[1], size[2]], False, 1, name='upsample11_m')
#         concat4_u11_c = ConcatLayer([upsample11_c, conv4], name='concat4_u11_c')
#         concat4_u11_m = ConcatLayer([upsample11_m, mask4], name='concat4_u11_m')
#         conv12, mask12 = P_Conv(concat4_u11_c, concat4_u11_m, n_filter=num_filter*8, name='pconv12')
#         # conv12 = BatchNormLayer(conv12, act=tf.nn.leaky_relu, is_train=istrain, name='bn12')
#
#         # # DPConv 5
#         size = conv3.outputs.get_shape().as_list()
#         upsample12_c = UpSampling2dLayer(conv12, [size[1], size[2]], False, 1, name='upsample12_c')
#         upsample12_m = UpSampling2dLayer(mask12, [size[1], size[2]], False, 1, name='upsample12_m')
#         # upsample12_c = UpSampling2dLayer(conv4, [size[1], size[2]], False, 1, name='upsample12_c')
#         # upsample12_m = UpSampling2dLayer(mask4, [size[1], size[2]], False, 1, name='upsample12_m')
#         concat3_u12_c = ConcatLayer([upsample12_c, conv3], name='concat3_u12_c')
#         concat3_u12_m = ConcatLayer([upsample12_m, mask3], name='concat3_u12_m')
#         conv13, mask13 = P_Conv(concat3_u12_c, concat3_u12_m, n_filter=num_filter*4, name='pconv13')
#         # conv13 = BatchNormLayer(conv13, act=tf.nn.leaky_relu, is_train=istrain, name='bn13')
#         # conv13 = InstanceNormLayer(conv13, act=tf.nn.elu, name='bn13')
#
#         # DPConv 6
#         size = conv2.outputs.get_shape().as_list()
#         upsample13_c = UpSampling2dLayer(conv13, [size[1], size[2]], False, 1, name='upsample13_c')
#         upsample13_m = UpSampling2dLayer(mask13, [size[1], size[2]], False, 1, name='upsample13_m')
#         # upsample13_c = UpSampling2dLayer(conv3, [size[1], size[2]], False, 1, name='upsample13_c')
#         # upsample13_m = UpSampling2dLayer(mask3, [size[1], size[2]], False, 1, name='upsample13_m')
#         concat2_u13_c = ConcatLayer([upsample13_c, conv2], name='concat2_u13_c')
#         concat2_u13_m = ConcatLayer([upsample13_m, mask2], name='concat2_u13_m')
#         conv14, mask14 = P_Conv(concat2_u13_c, concat2_u13_m, n_filter=num_filter*2, name='pconv14')
#         # conv14 = BatchNormLayer(conv14, act=tf.nn.leaky_relu, is_train=istrain, name='bn14')
#         # conv14 = InstanceNormLayer(conv14, act=tf.nn.elu, name='bn14')
#
#         # DPConv 7
#         size = conv1.outputs.get_shape().as_list()
#         upsample14_c = UpSampling2dLayer(conv14, [size[1], size[2]], False, 1, name='upsample14_c')
#         upsample14_m = UpSampling2dLayer(mask14, [size[1], size[2]], False, 1, name='upsample14_m')
#         # upsample14_c = UpSampling2dLayer(conv2, [size[1], size[2]], False, 1, name='upsample14_c')
#         # upsample14_m = UpSampling2dLayer(mask2, [size[1], size[2]], False, 1, name='upsample14_m')
#         concat1_u14_c = ConcatLayer([upsample14_c, conv1], name='concat1_u14_c')
#         concat1_u14_m = ConcatLayer([upsample14_m, mask1], name='concat1_u14_m')
#         conv15, mask15 = P_Conv(concat1_u14_c, concat1_u14_m, n_filter=num_filter, name='pconv15')
#         # conv15 = BatchNormLayer(conv15, act=tf.nn.leaky_relu, is_train=istrain, name='bn15')
#         # conv15 = InstanceNormLayer(conv15, act=tf.nn.elu, name='bn15')
#
#         # DPConv 8
#         size = n.outputs.get_shape().as_list()
#         upsample15_c = UpSampling2dLayer(conv15, [size[1], size[2]], False, 1, name='upsample15_c')
#         upsample15_m = UpSampling2dLayer(mask15, [size[1], size[2]], False, 1, name='upsample15_m')
#         # upsample15_c = UpSampling2dLayer(conv1, [size[1], size[2]], False, 1, name='upsample15_c')
#         # upsample15_m = UpSampling2dLayer(mask1, [size[1], size[2]], False, 1, name='upsample15_m')
#         concat0_u15_c = ConcatLayer([upsample15_c, n], name='concat0_u15_c')
#         concat0_u15_m = ConcatLayer([upsample15_m, m], name='concat0_u15_m')
#         conv16, mask16 = P_Conv(concat0_u15_c, concat0_u15_m, n_filter=1, name='pconv16')
#         # conv16.outputs = conv16.outputs*tf.abs(mask-1) + input
#
#         return conv16
    
# def Unetwork_nbn(input, mask, num_filter=8, istrain=False, reuse=False):
#
#     with tf.variable_scope('UNet', reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         n = InputLayer(input, name='depth')
#         m = InputLayer(mask, name='mask')
#
#         # Encoder Section
#         # PConv 1
#         # conv1_0, mask1_0 = P_Conv(n, m, g, n_filter=num_filter*2, filter_size=3, stride=1, name='pconv1_0')
#         # conv1_0, mask1_0 = P_Conv(conv1_0, mask1_0, g, n_filter=1, filter_size=7, stride=1, name='pconv1_1')
#         # conv1_0, mask1_0 = P_Conv(conv1_0, mask1_0, g, n_filter=1, filter_size=7, stride=1, name='pconv1_2')
#         # conv1_0, mask1_0 = P_Conv(conv1_0, mask1_0, g, n_filter=1, filter_size=7, stride=1, name='pconv1_3')
#         # conv1_0, mask1_0 = P_Conv(conv1_0, mask1_0, g, n_filter=1, filter_size=7, stride=1, name='pconv1_4')
#         # conv1_0, mask1_0 = P_Conv(conv1_0, mask1_0, g, n_filter=1, filter_size=7, stride=1, name='pconv1_5')
#         conv1, mask1 = P_Conv(n, m, n_filter=num_filter, filter_size=3, stride=2, name='pconv1')
#         # all_paras.append(conv1.all_params)
#         # conv1, mask1 = P_Conv(n, m, n_filter=num_filter, filter_size=3, stride=1, name='pconv1_1')
#         # conv1, mask1 = P_Conv(conv1, mask1, n_filter=num_filter, filter_size=3, stride=1, name='pconv1_2')
#         # conv1, mask1 = P_Conv(conv1, mask1, n_filter=1, filter_size=3, stride=1, name='pconv1_3')
#
#         # PConv 2
#         conv2, mask2 = P_Conv(conv1, mask1, filter_size=3, n_filter=num_filter, stride=1, name='pconv2_0')
#         conv2,  mask2 = P_Conv(conv2, mask2, filter_size=3, n_filter=num_filter * 2, stride=2, name='pconv2')
#         # conv2 = BatchNormLayer(conv2, act=tf.nn.leaky_relu, is_train=istrain, name='bn2')
#         # conv2 = InstanceNormLayer(conv2, act=tf.nn.elu, name='bn2')
#
#         #  PConv 3
#         conv3, mask3 = P_Conv(conv2, mask2, n_filter=num_filter * 2, filter_size=3, stride=1, name='pconv3_0')
#         conv3, mask3 = P_Conv(conv3, mask3, n_filter=num_filter * 4, filter_size=3, stride=2, name='pconv3')
#         # conv3 = BatchNormLayer(conv3, act=tf.nn.leaky_relu, is_train=istrain, name='bn3')
#         # conv3 = InstanceNormLayer(conv3, act=tf.nn.elu, name='bn3')
#
#         # # PConv 4
#         conv4,  mask4 = P_Conv(conv3, mask3, n_filter=num_filter * 4, filter_size=3, stride=1, name='pconv4')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 16, filter_size=3, stride=2, name='pconv4_1')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 16, filter_size=3, stride=1, name='pconv4_2')
#         conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=3, stride=1, name='pconv4_3')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_4')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_5')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_6')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_7')
#         # conv4, mask4 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=5, stride=1, name='pconv4_8')
#         # conv4 = BatchNormLayer(conv4, act=tf.nn.leaky_relu, is_train=istrain, name='bn4')
#         # conv4 = InstanceNormLayer(conv4, act=tf.nn.elu, name='bn4')
#
#         # PConv 5
#         conv5, mask5 = P_Conv(conv4, mask4, n_filter=num_filter * 8, filter_size=3, stride=1, name='pconv5')
#         conv5, mask5 = P_Conv(conv5, mask5, n_filter=num_filter * 16, filter_size=3, stride=2, name='pconv5_1')
#         conv5, mask5 = P_Conv(conv5, mask5, n_filter=num_filter * 16, filter_size=3, stride=1, name='pconv5_2')
#         conv5, mask5 = P_Conv(conv5, mask5, n_filter=num_filter * 16, filter_size=3, stride=1, name='pconv5_3')
#         # conv5 = BatchNormLayer(conv5, act=tf.nn.leaky_relu, is_train=istrain, name='bn5')
#
#         # # PConv 6
#         # conv6,  mask6 = P_Conv(conv5, mask5, n_filter=num_filter*8, filter_size=5, stride=2, name='pconv6')
#         # conv6 = BatchNormLayer(conv6, act=tf.nn.leaky_relu, is_train=istrain, name='bn6')
#
#         # # PConv 7
#         # # conv7, mask7 = P_Conv(conv6, mask6, n_filter=num_filter*8, filter_size=3, stride=2, name='pconv7')
#         # # conv7 = BatchNormLayer(conv7, act=tf.nn.relu, is_train=istrain, name='bn7')
#
#         # # # PConv 8
#         # # conv8,  mask8 = P_Conv(conv7, mask7, n_filter=num_filter*8, filter_size=3, stride=2, name='pconv8')
#         # # conv8 = BatchNormLayer(conv8, act=tf.nn.relu, is_train=istrain, name='bn8')
#
#         # # Decoder Section
#         # # # DPConv 1
#         # # size = conv7.outputs.get_shape().as_list()
#         # # upsample8_c = UpSampling2dLayer(conv8, [size[1], size[2]], False, 1, name='upsample8_c')
#         # # upsample8_m = UpSampling2dLayer(mask8, [size[1], size[2]], False, 1, name='upsample8_m')
#         # # concat7_u8_c = ConcatLayer([upsample8_c, conv7], name='concat7_u8_c')
#         # # concat7_u8_m = ConcatLayer([upsample8_m, mask7], name='concat7_u8_m')
#         # # conv9, mask9 = P_Conv(concat7_u8_c, concat7_u8_m, n_filter=num_filter*8, name='pconv9')
#         # # conv9 = BatchNormLayer(conv9, act=tf.nn.leaky_relu, is_train=istrain, name='bn9')
#
#         # # # DPConv 2
#         # size = conv6.outputs.get_shape().as_list()
#         # upsample9_c = UpSampling2dLayer(conv7, [size[1], size[2]], False, 1, name='upsample9_c')
#         # upsample9_m = UpSampling2dLayer(mask7, [size[1], size[2]], False, 1, name='upsample9_m')
#         # concat6_u9_c = ConcatLayer([upsample9_c, conv6], name='concat6_u9_c')
#         # concat6_u9_m = ConcatLayer([upsample9_m, mask6], name='concat6_u9_m')
#         # conv10, mask10 = P_Conv(concat6_u9_c, concat6_u9_m, n_filter=num_filter*8, name='pconv10')
#         # # conv10 = BatchNormLayer(conv10, act=tf.nn.leaky_relu, is_train=istrain, name='bn10')
#
#         # # DPConv 3
#         # size = conv5.outputs.get_shape().as_list()
#         # # upsample10_c = UpSampling2dLayer(conv10, [size[1], size[2]], False, 1, name='upsample10_c')
#         # # upsample10_m = UpSampling2dLayer(mask10, [size[1], size[2]], False, 1, name='upsample10_m')
#         # upsample10_c = UpSampling2dLayer(conv6, [size[1], size[2]], False, 1, name='upsample10_c')
#         # upsample10_m = UpSampling2dLayer(mask6, [size[1], size[2]], False, 1, name='upsample10_m')
#         # concat5_u10_c = ConcatLayer([upsample10_c, conv5], name='concat5_u10_c')
#         # concat5_u10_m = ConcatLayer([upsample10_m, mask5], name='concat5_u10_m')
#         # conv11, mask11 = P_Conv(concat5_u10_c, concat5_u10_m, n_filter=num_filter*8, name='pconv11')
#         # conv11 = BatchNormLayer(conv11, act=tf.nn.leaky_relu, is_train=istrain, name='bn11')
#
#         # DPConv 4
#         size = conv4.outputs.get_shape().as_list()
#         # upsample11_c = UpSampling2dLayer(conv11, [size[1], size[2]], False, 1, name='upsample11_c')
#         # upsample11_m = UpSampling2dLayer(mask11, [size[1], size[2]], False, 1, name='upsample11_m')
#         upsample11_c = UpSampling2dLayer(conv5, [size[1], size[2]], False, 1, name='upsample11_c')
#         upsample11_m = UpSampling2dLayer(mask5, [size[1], size[2]], False, 1, name='upsample11_m')
#         concat4_u11_c = ConcatLayer([upsample11_c, conv4], name='concat4_u11_c')
#         concat4_u11_m = ConcatLayer([upsample11_m, mask4], name='concat4_u11_m')
#         conv12, mask12 = P_Conv(concat4_u11_c, concat4_u11_m, n_filter=num_filter*8, name='pconv12')
#         # conv12 = BatchNormLayer(conv12, act=tf.nn.leaky_relu, is_train=istrain, name='bn12')
#
#         # # DPConv 5
#         size = conv3.outputs.get_shape().as_list()
#         upsample12_c = UpSampling2dLayer(conv12, [size[1], size[2]], False, 1, name='upsample12_c')
#         upsample12_m = UpSampling2dLayer(mask12, [size[1], size[2]], False, 1, name='upsample12_m')
#         # upsample12_c = UpSampling2dLayer(conv4, [size[1], size[2]], False, 1, name='upsample12_c')
#         # upsample12_m = UpSampling2dLayer(mask4, [size[1], size[2]], False, 1, name='upsample12_m')
#         concat3_u12_c = ConcatLayer([upsample12_c, conv3], name='concat3_u12_c')
#         concat3_u12_m = ConcatLayer([upsample12_m, mask3], name='concat3_u12_m')
#         conv13, mask13 = P_Conv(concat3_u12_c, concat3_u12_m, n_filter=num_filter*4, name='pconv13')
#         # conv13 = BatchNormLayer(conv13, act=tf.nn.leaky_relu, is_train=istrain, name='bn13')
#         # conv13 = InstanceNormLayer(conv13, act=tf.nn.elu, name='bn13')
#
#         # DPConv 6
#         size = conv2.outputs.get_shape().as_list()
#         upsample13_c = UpSampling2dLayer(conv13, [size[1], size[2]], False, 1, name='upsample13_c')
#         upsample13_m = UpSampling2dLayer(mask13, [size[1], size[2]], False, 1, name='upsample13_m')
#         # upsample13_c = UpSampling2dLayer(conv3, [size[1], size[2]], False, 1, name='upsample13_c')
#         # upsample13_m = UpSampling2dLayer(mask3, [size[1], size[2]], False, 1, name='upsample13_m')
#         concat2_u13_c = ConcatLayer([upsample13_c, conv2], name='concat2_u13_c')
#         concat2_u13_m = ConcatLayer([upsample13_m, mask2], name='concat2_u13_m')
#         conv14, mask14 = P_Conv(concat2_u13_c, concat2_u13_m, n_filter=num_filter*2, name='pconv14')
#         # conv14 = BatchNormLayer(conv14, act=tf.nn.leaky_relu, is_train=istrain, name='bn14')
#         # conv14 = InstanceNormLayer(conv14, act=tf.nn.elu, name='bn14')
#
#         # DPConv 7
#         size = conv1.outputs.get_shape().as_list()
#         upsample14_c = UpSampling2dLayer(conv14, [size[1], size[2]], False, 1, name='upsample14_c')
#         upsample14_m = UpSampling2dLayer(mask14, [size[1], size[2]], False, 1, name='upsample14_m')
#         # upsample14_c = UpSampling2dLayer(conv2, [size[1], size[2]], False, 1, name='upsample14_c')
#         # upsample14_m = UpSampling2dLayer(mask2, [size[1], size[2]], False, 1, name='upsample14_m')
#         concat1_u14_c = ConcatLayer([upsample14_c, conv1], name='concat1_u14_c')
#         concat1_u14_m = ConcatLayer([upsample14_m, mask1], name='concat1_u14_m')
#         conv15, mask15 = P_Conv(concat1_u14_c, concat1_u14_m, n_filter=num_filter, name='pconv15')
#         # conv15 = BatchNormLayer(conv15, act=tf.nn.leaky_relu, is_train=istrain, name='bn15')
#         # conv15 = InstanceNormLayer(conv15, act=tf.nn.elu, name='bn15')
#
#         # DPConv 8
#         size = n.outputs.get_shape().as_list()
#         upsample15_c = UpSampling2dLayer(conv15, [size[1], size[2]], False, 1, name='upsample15_c')
#         upsample15_m = UpSampling2dLayer(mask15, [size[1], size[2]], False, 1, name='upsample15_m')
#         # upsample15_c = UpSampling2dLayer(conv1, [size[1], size[2]], False, 1, name='upsample15_c')
#         # upsample15_m = UpSampling2dLayer(mask1, [size[1], size[2]], False, 1, name='upsample15_m')
#         concat0_u15_c = ConcatLayer([upsample15_c, n], name='concat0_u15_c')
#         concat0_u15_m = ConcatLayer([upsample15_m, m], name='concat0_u15_m')
#         conv16, mask16 = P_Conv(concat0_u15_c, concat0_u15_m, n_filter=1, name='pconv16')
#
#         return conv16