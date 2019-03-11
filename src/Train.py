#!usr/bin/env/ python
# _*_ coding:utf-8 _*_
import tensorflow as tf
from StereoNet import StereoNet
import ImgUtils
import numpy as np
import cv2 as cv
import time
import random

imagenet_stats = [[0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225]]

n_epochs = 10
batch_size = 1

width = 960
height = 540

def data_batch_get(x_train, y_train, iteration, batch_size, pfm):
    imgL = []
    imgR = []
    disp_true = []
    for i in range(batch_size):
        try:
            imgL.append(cv.imread(x_train[iteration * batch_size * 2 + i]))
            if pfm == True:
                disp_true.append(ImgUtils.readPFM(y_train[iteration])[0])
            else:
                disp_true.append(cv.imread(y_train[iteration], cv.IMREAD_GRAYSCALE))
        except:
            print(x_train[iteration * batch_size * 2])
            print(y_train[iteration * batch_size * 2 + batch_size])
            return None
    for i in range(batch_size):
        imgR.append(cv.imread(x_train[iteration * batch_size * 2 + batch_size + i]))

    imgL = (np.array(imgL)).astype(np.float32) / 255.0

    imgR = (np.array(imgR)).astype(np.float32) / 255.0
    disp_true = (np.array(disp_true)).astype(np.float32)

    return imgL, imgR, disp_true
    
    
def train_data_batch_get(x_train, y_train, iteration, batch_size, pfm):
    imgL = []
    imgR = []
    disp_true = []

    w, h = width, height
    th, tw = 256, 512

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for i in range(batch_size):
        try:
            imgL.append(cv.imread(x_train[iteration * batch_size * 2 + i])[y1:y1+th, x1:x1+tw, :])
            if pfm == True:
                disp_true.append(ImgUtils.readPFM(y_train[iteration])[0][y1:y1+th, x1:x1+tw])
            else:
                disp_true.append(cv.imread(y_train[iteration], cv.IMREAD_GRAYSCALE))
        except:
            print(x_train[iteration * batch_size * 2])
            print(y_train[iteration * batch_size * 2 + batch_size])
            return None
    for i in range(batch_size):
        imgR.append(cv.imread(x_train[iteration * batch_size * 2 + batch_size + i])[y1:y1+th, x1:x1+tw, :])

    imgL = (np.array(imgL)).astype(np.float32) / 255.0

    imgR = (np.array(imgR)).astype(np.float32) / 255.0
    disp_true = (np.array(disp_true)).astype(np.float32)

    return imgL, imgR, disp_true


def pre_process(image_data, channels, batch_size):
    for i in range(batch_size):
        for j in range(channels):
            image_data[i, :, :, j] = (image_data[i, :, :, j] - imagenet_stats[0][j]) /  imagenet_stats[1][j]

    return image_data

class Network():
    def __init__(self,
                 height=375,
                 width=1242,
                 channel=3,
                 batch_size=1,
                 learning_rate=1e-3):
        self.height = height
        self.width = width
        self.channel = channel
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        
    def loss(self, modelOutput, groundTruth):
        '''
        temp1 = tf.square(groundTruth - modelOutput)
        temp2 = tf.div(temp1, 4.0)
        temp3 = tf.add(temp2, 1.0)
        temp4 = tf.sqrt(temp3)
        temp5 = tf.subtract(temp4, 1.0)

        loss_all = tf.reduce_sum(temp5)
        '''
        diff = tf.abs(groundTruth - modelOutput)
        less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
        loss_all = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
        loss_all = tf.reduce_sum(loss_all)
        
        return loss_all, groundTruth, modelOutput

        
    def train_model(self, loss, weight_decay=0.0001):
        with tf.name_scope('train'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=weight_decay)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                training_op = optimizer.minimize(loss)
            return training_op

            
    def get_placeholder(self):
        x_left = tf.placeholder(tf.float32, shape=(self.batch_size, self.height, self.width, self.channel), name='x_left')
        x_right = tf.placeholder(tf.float32, shape=(self.batch_size, self.height, self.width, self.channel), name='x_right')
        y = tf.placeholder(tf.float32, shape=(self.batch_size, self.height, self.width), name='y')
        is_training = tf.placeholder(tf.bool, shape=(None), name='is_training')
        
        return x_left, x_right, y, is_training

        
x_train = ImgUtils.GetAllTrainFile(r'F:\Intership\sceneFlow\frames_cleanpass', 'rgb')
y_train = ImgUtils.GetAllTrainFile(r'F:\Intership\sceneFlow\disparity', 'disp')
  
tf.reset_default_graph()

#x_train = ImgUtils.Flying3DGetAllTrainFile(r'F:\Intership\FlyingThings3D_subset_image_clean\FlyingThings3D_subset\train\image_clean', 'rgb')
#y_train = ImgUtils.Flying3DGetAllTrainFile(r'F:\Intership\FlyingThings3D_subset_disparity\FlyingThings3D_subset\train\disparity', 'disp')

random_list = [i * 2 for i in range(len(y_train))]
random.shuffle(random_list)

new_x_train = []
new_y_train = []
for i in random_list:
    new_x_train.append(x_train[i])
    new_x_train.append(x_train[i + 1])
    new_y_train.append(y_train[i // 2])

with tf.Session() as sess:
    train_h = 256
    train_w = 512
    net = Network(height=train_h,
                  width=train_w,
                  batch_size=batch_size,
                  learning_rate=1e-3)
    x_left, x_right, y, is_training = net.get_placeholder()
    model = StereoNet(batch_size, x_left, x_right, train_h, train_w, is_training)
    outputs = model.forward()
    loss_op, m1, m2 = net.loss(outputs, y)
    training_op = net.train_model(loss_op)

    log_dir = r'./summary'
    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', loss_op)
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(log_dir + '/train' + time.strftime("%Y%m%d%H%M%S", time.localtime()), sess.graph)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    init.run()

    for epoch in range(n_epochs):
        train_len = len(x_train)
        for iteration in range(train_len // (batch_size * 2)):
            x_left_batch, x_right_batch, y_batch = train_data_batch_get(new_x_train, new_y_train, iteration, batch_size,
                                                                  pfm=True)
            x_left_batch = pre_process(x_left_batch, 3, batch_size)
            x_right_batch = pre_process(x_right_batch, 3, batch_size)

            _, loss_val, merged_summary = sess.run([training_op, loss_op, merged], feed_dict={x_left: x_left_batch, x_right: x_right_batch, y: y_batch,
                                             is_training: True})
            train_writer.add_summary(merged_summary, iteration)

            
            if iteration % 500 == 0:
                m1Val = m1.eval(feed_dict={x_left: x_left_batch, x_right: x_right_batch, y: y_batch,
                                                is_training: True})
                #secondVal = second.eval(feed_dict={x_left: x_left_batch, x_right: x_right_batch, y: y_batch,
                #                                is_training: True})
                pic = outputs.eval(feed_dict={x_left: x_left_batch, x_right: x_right_batch, y: y_batch,
                                                is_training: True})

                cv.imshow("x_left_batchpic", x_left_batch.reshape(train_h, train_w, -1))
                cv.imshow("x_right_batchpic", x_right_batch.reshape(train_h, train_w, -1))

                cv.imshow("pic", m1Val.reshape(train_h, train_w, -1).astype(np.uint8))
                #cv.imshow("secondVal", secondVal.reshape(height, width, -1))
                cv.imshow("pic3", pic.reshape(train_h, train_w, -1).astype(np.uint8))
                cv.waitKey(0)

            print("epoch:", epoch, "  iteration:", iteration, "\tloss:", loss_val)
            '''
            if(iteration % 10 == 0):
                pic = outputs.eval(feed_dict={x: x_batch, y: y_batch, is_training: True})
                cv.imshow("pic", pic.reshape(375, 1242, -1))
                cv.imshow("pic2", y_batch.reshape(375, 1242, -1))
                cv.waitKey(0)
            '''
        saver.save(sess, './checkpoint_dir/MyModel')

    train_writer.close()

