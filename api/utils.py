import os
import numpy as np
import cv2


class Vector(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])

    def get_vec(self):
        return self.vec

    def length(self):
        return np.linalg.norm(self.vec)


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_middle(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def get_vector_angle(v1, v2):
    _cos = np.dot(v1.get_vec(), v2.get_vec()) / (v1.length() * v2.length())
    _sin = np.cross(v1.get_vec(), v2.get_vec()) / (v1.length() * v2.length())
    return np.arctan2(_sin, _cos)


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated