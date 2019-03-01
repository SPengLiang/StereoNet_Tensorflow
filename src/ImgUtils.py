#!usr/bin/env/ python
# _*_ coding:utf-8 _*_
import re
import numpy as np
import os

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def GetAllTrainFile(root_path, mode):
    all_train_name = []

    firstDir = os.listdir(root_path)
    for name in firstDir:
        dir1 = os.path.join(root_path, name)
        secondDir = os.listdir(dir1)
        for second_name in secondDir:
            dir2 = os.path.join(dir1, second_name)
            thirdDir = os.listdir(dir2)
            for third_name in thirdDir:
                dir3 = os.path.join(dir2, third_name)
                fourthDir = os.listdir(dir3)

                if mode == 'rgb':
                    left_name = []
                    right_name = []

                    for fourth_name in fourthDir:
                        if fourth_name == 'left':
                            dir4 = os.path.join(dir3, fourth_name)
                            finalDir = os.listdir(dir4)
                            for file_name in finalDir:
                                left_name.append(os.path.join(dir4, file_name))
                        else:
                            dir4 = os.path.join(dir3, fourth_name)
                            finalDir = os.listdir(dir4)
                            for file_name in finalDir:
                                right_name.append(os.path.join(dir4, file_name))
                    for i in range(len(left_name)):
                        all_train_name.append(left_name[i])
                        all_train_name.append(right_name[i])

                else:
                    for fourth_name in fourthDir:
                        if fourth_name == 'left':
                            dir4 = os.path.join(dir3, fourth_name)
                            finalDir = os.listdir(dir4)
                            for file_name in finalDir:
                                all_train_name.append(os.path.join(dir4, file_name))

    return all_train_name


def Flying3DGetAllTrainFile(root_path, mode):
    all_train_name = []

    fourthDir = os.listdir(root_path)

    if mode == 'rgb':
        left_name = []
        right_name = []

        for fourth_name in fourthDir:
            if fourth_name == 'left':
                dir4 = os.path.join(root_path, fourth_name)
                finalDir = os.listdir(dir4)
                for file_name in finalDir:
                    left_name.append(os.path.join(dir4, file_name))
            else:
                dir4 = os.path.join(root_path, fourth_name)
                finalDir = os.listdir(dir4)
                for file_name in finalDir:
                    right_name.append(os.path.join(dir4, file_name))
        for i in range(len(left_name)):
            all_train_name.append(left_name[i])
            all_train_name.append(right_name[i])

    else:
        for fourth_name in fourthDir:
            if fourth_name == 'left':
                dir4 = os.path.join(root_path, fourth_name)
                finalDir = os.listdir(dir4)
                for file_name in finalDir:
                    all_train_name.append(os.path.join(dir4, file_name))

    return all_train_name
