#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   api\scence.py 
@Time    :   2022-01-18 12:17:38 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import os
import cv2
import numpy as np
from api.utils import *


class Scence(object):
    def __init__(self, args):
        self.args = args
        self.background = cv2.imread(args.background, cv2.IMREAD_UNCHANGED)
        # self.background = cv2.resize(self.background, (1920, 1080))
        self.background = cv2.cvtColor(self.background, cv2.COLOR_BGR2RGB)
        self.img = self.background.copy()

    def merge_img(self, img, point):
        # print(img.shape, self.img.shape)
        alpha = img[:, :, 3] / 255.0
        origin_alpha = 1 - alpha
        point = [point[1], point[0]]

        if point[0] - img.shape[0] / 2 < 0:
            point[0] = img.shape[0] / 2

        if point[1] - img.shape[1] / 2 < 0:
            point[1] = img.shape[1] / 2

        if point[0] + img.shape[0] / 2 > self.img.shape[0]:
            point[0] = self.img.shape[0] - img.shape[0] / 2

        if point[1] + img.shape[1] / 2 > self.img.shape[1]:
            point[1] = self.img.shape[1] - img.shape[1] / 2


        self.img[int(point[0] - img.shape[0] / 2):int(point[0] + img.shape[0] / 2),
                    int(point[1] - img.shape[1] / 2):int(point[1] + img.shape[1] / 2), :] = \
            img[:, :, :3] * alpha[:, :, np.newaxis] + self.img[
                int(point[0] - img.shape[0] / 2):int(point[0] + img.shape[0] / 2),
                int(point[1] - img.shape[1] / 2):int(point[1] + img.shape[1] / 2), :] * origin_alpha[:, :, np.newaxis]

    def draw_emoji(self, point, size, emoji_id, rot=0, hor_flip=False):
        print(point, size, emoji_id)
        emoji = os.path.join('assert', 'emoji', '{}.png'.format(emoji_id.strip(':')))
        emoji = cv2.imread(emoji, cv2.IMREAD_UNCHANGED)
        if hor_flip:
            # horizontal flip
            emoji = cv2.flip(emoji, 1)
        # print(emoji.shape)
        emoji = cv2.resize(emoji, (size, size))
        # rotate
        emoji = rotate(emoji, rot * 180 / np.pi)
        # emoji = cv2.cvtColor(emoji, cv2.COLOR_BGR2RGB)
        self.merge_img(emoji, point)

    def get_scence(self, lmList):
        self.img = self.background.copy()
        if lmList is None or len(lmList) == 0:
            return self.img

        pose_kps = {}

        for item in lmList:
            pose_kps[item[0]] = (item[1], item[2])

        # shorts
        self.draw_emoji(get_middle(
            get_middle(pose_kps[24], pose_kps[23]),
            get_middle(get_middle(pose_kps[24], pose_kps[23]),
                       get_middle(pose_kps[26], pose_kps[25]))),
                        192,
                        ':shorts:',
                        rot=get_vector_angle(Vector(pose_kps[24], pose_kps[23]),
                                             Vector((0, 0), (1, 0))),
                        hor_flip=False)

        # draw emoji face
        # self.draw_emoji(pose_kps[0], int(np.ceil(distance(pose_kps[2], pose_kps[3]) * 2.5 * 5)),
        #                 ':dog:')
        # face
        self.draw_emoji(pose_kps[0], 192, ':hot_face:')
        # left hand
        self.draw_emoji(pose_kps[16],
                        128,
                        ':hand:',
                        rot=get_vector_angle(
                            Vector(get_middle(pose_kps[20], pose_kps[18]), pose_kps[16]),
                            Vector((0, 0), (0, 1))))
        # right hand
        self.draw_emoji(pose_kps[15],
                        128,
                        ':hand:',
                        rot=get_vector_angle(
                            Vector(get_middle(pose_kps[20], pose_kps[18]), pose_kps[16]),
                            Vector((0, 0), (0, 1))),
                        hor_flip=True)

        # left foot
        self.draw_emoji(get_middle(pose_kps[30], pose_kps[32]),
                        128,
                        ':shoe:',
                        rot=get_vector_angle(Vector(pose_kps[30], pose_kps[32]),
                                             Vector((0, 0), (-1, 0))),
                        hor_flip=True)
        # right foot
        self.draw_emoji(get_middle(pose_kps[29], pose_kps[31]),
                        128,
                        ':shoe:',
                        rot=get_vector_angle(Vector(pose_kps[29], pose_kps[31]),
                                             Vector((0, 0), (1, 0))),
                        hor_flip=False)
        return self.img

    def save_scence(self, filename):
        cv2.imwrite(filename, self.img)