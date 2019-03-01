#!usr/bin/env/ python
# _*_ coding:utf-8 _*_

import tensorflow as tf
import tensorflow.contrib.slim as slim

class StereoNet():
    def __init__(self, batchSize, left_img, right_img, height, width, is_training):
        self.left_img = left_img
        self.right_img = right_img
        self.height = height
        self.width = width
        self.batchSize = batchSize
        self.is_training = is_training


    def down_sample(self, input):
        with tf.name_scope('DownSample'):
            downSample1 = slim.conv2d(input, 32, [5, 5], 2, 'SAME', activation_fn=None)
            downSample2 = slim.conv2d(downSample1, 32, [5, 5], 2, 'SAME', activation_fn=None)
            downSample3 = slim.conv2d(downSample2, 32, [5, 5], 2, 'SAME', activation_fn=None)
            downSample4 = slim.conv2d(downSample3, 32, [5, 5], 2, 'SAME', activation_fn=None)
        return downSample4


    def first_res_block(self, input):
        with tf.name_scope('FirstResBolck'):
            res1 = res_bolck(input, 32, [3, 3], self.is_training)
            res2 = res_bolck(res1, 32, [3, 3], self.is_training)
            res3 = res_bolck(res2, 32, [3, 3], self.is_training)
            res4 = res_bolck(res3, 32, [3, 3], self.is_training)
            res5 = res_bolck(res4, 32, [3, 3], self.is_training)
            res6 = res_bolck(res5, 32, [3, 3], self.is_training)
        output = slim.conv2d(res6, 32, [3, 3], activation_fn=None)
        return output


    def cost_volume(self, left_feature, right_feature):
        cost_aggre = cost_volume_aggre(left_feature, right_feature, 4, 192)
        with tf.name_scope('CostVolume'):
            cost_volume1 = conv3d_bolck(cost_aggre, 32, [3, 3, 3], self.is_training)
            cost_volume2 = conv3d_bolck(cost_volume1, 32, [3, 3, 3], self.is_training)
            cost_volume3 = conv3d_bolck(cost_volume2, 32, [3, 3, 3], self.is_training)
            cost_volume4 = conv3d_bolck(cost_volume3, 32, [3, 3, 3], self.is_training)
        output = slim.conv3d(cost_volume4, 1, [3, 3, 3], padding='SAME', activation_fn=None)

        return tf.squeeze(output, 4)


    def second_res_block(self, input):
        pre_change_channel = slim.conv2d(input, 32, [3, 3] )
        res1 = atrous_res_bolck(pre_change_channel, [3, 3, 32, 32], 1, self.is_training)
        res2 = atrous_res_bolck(res1, [3, 3, 32, 32], 2, self.is_training)
        res3 = atrous_res_bolck(res2, [3, 3, 32, 32], 4, self.is_training)
        res4 = atrous_res_bolck(res3, [3, 3, 32, 32], 8, self.is_training)
        res5 = atrous_res_bolck(res4, [3, 3, 32, 32], 1, self.is_training)
        res6 = atrous_res_bolck(res5, [3, 3, 32, 32], 1, self.is_training)
        output = slim.conv2d(res6, 1, [3, 3], activation_fn=None)

        return output


    def forward(self):
        left_downsample_output = self.down_sample(self.left_img)
        right_downsample_output = self.down_sample(self.right_img)

        left_first_resblock_output = self.first_res_block(left_downsample_output)
        right_first_resblock_output = self.first_res_block(right_downsample_output)

        cost_volume_output = self.cost_volume(left_first_resblock_output, right_first_resblock_output)

        corase_disp = soft_arg_min(cost_volume_output)

        # 双线性插值恢复到原图尺寸
        corase_disp = tf.reshape(corase_disp,
                                [corase_disp.shape[0], corase_disp.shape[1], corase_disp.shape[2], 1])
        new_corase_disp = tf.image.resize_images(corase_disp, [self.height, self.width], align_corners=True)

        new_input = tf.concat([self.left_img, new_corase_disp], 3)

        second_res_block_output = self.second_res_block(new_input)

        finalOutput = new_corase_disp + second_res_block_output
        finalOutput = tf.nn.relu(tf.squeeze(finalOutput, 3))

        return finalOutput


def res_bolck(org, channel, kernelSize, is_training):
    conv1 = slim.conv2d(org, channel, kernelSize, activation_fn=None)
    bn1 = slim.batch_norm(conv1, is_training=is_training)
    leaky1 = tf.nn.leaky_relu(bn1, alpha=0.2)

    conv2 = slim.conv2d(leaky1, channel, kernelSize, activation_fn=None)
    bn2 = slim.batch_norm(conv2, is_training=is_training)
    out= org + bn2
    out = tf.nn.leaky_relu(out, alpha=0.2)

    return out

def cost_volume_aggre(img_L, img_R, k, max_disp):
    d = max_disp // (2**k) - 1
    dp_list = []

    #disparity is 0
    elw_tf = tf.concat([img_L, img_R], 3)
    dp_list.append(elw_tf)

    #disparity is not 0
    for dis in range(d):
        # moving the features by disparity d can be done by padding zeros
        pad = tf.constant([[0, 0], [0, 0], [dis + 1, 0], [0, 0]], dtype=tf.int32)
        pad_R = tf.pad(img_R[:, :, :-1 - dis, :], pad, "CONSTANT")
        elw_tf = tf.concat([img_L, pad_R], 3)
        dp_list.append(elw_tf)

    total_pack_tf = tf.concat(dp_list, 0)
    total_pack_tf = tf.expand_dims(total_pack_tf, 0)
    return total_pack_tf


def conv3d_bolck(org, channel, kernelSize, is_training):
    conv = slim.conv3d(org, channel, kernelSize, padding='SAME', activation_fn=None)
    cnn3d_bn = tf.contrib.layers.batch_norm(
                                            conv,
                                            data_format='NHWC',  # Matching the "cnn" tensor which has shape (?, 9, 120, 160, 96).
                                            center=True,
                                            scale=True,
                                            is_training=is_training)
    leaky = tf.nn.leaky_relu(cnn3d_bn, alpha=0.2)
    return leaky


def soft_arg_min(input):

    negDispCost = input
    dispWeight = tf.nn.softmax(negDispCost, axis=1)
    d_grid = tf.cast(tf.range(0, input.shape[1].value), dtype=tf.float32)

    d_grid = tf.reshape(d_grid, [1, -1, 1, 1])

    d_grid = tf.tile(d_grid, [input.shape[0].value, 1, input.shape[2].value, input.shape[3].value])#tf.tile(d_grid, [2])

    tmp = dispWeight * d_grid
    arg_soft_min = tf.reduce_sum(tmp, axis=1)
    '''
    trans = tf.transpose(input, perm=[0, 2, 3, 1])
    neg = tf.negative(trans)
    logits = tf.nn.softmax(neg)

    disparity_filter = tf.reshape(tf.range(0, input.shape[1].value, 1, dtype=tf.float32), [1, 1, input.shape[1].value, 1])
    distrib = tf.nn.conv2d(logits, disparity_filter, strides=[1, 1, 1, 1], padding='SAME')
    '''
    return arg_soft_min


def atrous_res_bolck(org, kernelSize, rate, is_training):
    conv1 = slim.conv2d(org, kernelSize[-1], [3, 3], padding='SAME', rate=[rate, rate], activation_fn=None)
    bn1 = slim.batch_norm(conv1, is_training=is_training)
    leaky1 = tf.nn.leaky_relu(bn1, alpha=0.2)

    conv2 = slim.conv2d(leaky1, kernelSize[-1], [3, 3], padding='SAME', rate=[rate, rate], activation_fn=None)
    bn2 = slim.batch_norm(conv2, is_training=is_training)
    out= org + bn2
    out = tf.nn.leaky_relu(out, alpha=0.2)
    return out
