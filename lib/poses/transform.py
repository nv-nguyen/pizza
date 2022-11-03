# https://github.com/lvsn/6DOF_tracking_evaluation/blob/master/ulaval_6dof_object_tracking/utils/transform.py
"""
    Transforms contain utility functions to manipulate pointclouds
    date : 2016-03-01
"""
__author__ = "Mathieu Garon"
__version__ = "0.0.1"

import numpy as np
import random
import math
from lib.poses.utils import rodrigues, rodrigues_inverse, euler2quat, mat2xyz_euler
from scipy.spatial.transform import Rotation as R


class Transform:
    def __init__(self):
        self.matrix = np.eye(4, dtype=np.float32)

    def set_translation(self, x, y, z):
        self.matrix[0:3, 3] = [x, y, z]

    def set_rotation(self, x, y, z):
        self.matrix[0:3, 0:3] = rodrigues(x, y, z)

    def translate(self, x=0, y=0, z=0, transform=None):
        """
        The translation is applied before the transforme
        TODO: this is not consistant...
        :param x:
        :param y:
        :param z:
        :param transform:
        :return:
        """
        if transform:
            new_transform = transform.copy()
        else:
            new_transform = Transform.from_parameters(x, y, z, 0, 0, 0)
        new_transform.combine(self)
        self.matrix = new_transform.matrix
        return self

    def rotate(self, x=0, y=0, z=0, transform=None):
        if transform:
            self.combine(transform)
        else:
            self.combine(Transform.from_parameters(0, 0, 0, x, y, z))
        return self

    @staticmethod
    def random(translation_range=(-1, 1), rotation_range=(-1, 1)):
        x = random.uniform(*translation_range)
        y = random.uniform(*translation_range)
        z = random.uniform(*translation_range)
        a = random.uniform(*rotation_range)
        b = random.uniform(*rotation_range)
        c = random.uniform(*rotation_range)
        return Transform.from_parameters(x, y, z, a, b, c)

    @staticmethod
    def from_matrix(matrix):
        ret = Transform()
        ret.matrix = matrix
        return ret

    @staticmethod
    def scale(x, y, z):
        ret = Transform()
        ret.matrix[0, 0] = x
        ret.matrix[1, 1] = y
        ret.matrix[2, 2] = z
        return ret

    @staticmethod
    def lookAt(eye, center, up):
        ret = Transform()
        E = eye
        C = center
        U = up

        F = C - E
        F /= np.linalg.norm(F)
        S = np.cross(F, U)
        S /= np.linalg.norm(S)
        U = np.cross(S, F)

        mat = np.eye(4, dtype=np.float32)

        mat[0, :] = np.hstack([S, 0])
        mat[1, :] = np.hstack([U, 0])
        mat[2, :] = np.hstack([-F, 0])
        mat[0, 3] = -np.dot(S, E)
        mat[1, 3] = -np.dot(U, E)
        mat[2, 3] = np.dot(F, E)
        ret.matrix = mat
        return ret

    def to_parameters(self, isDegree=False, isQuaternion=False, rodrigues=True):
        x, y, z = self.matrix[0:3, 3]
        if rodrigues:
            rx, ry, rz = rodrigues_inverse(self.matrix[0:3, 0:3])
        else:
            rx, ry, rz = mat2xyz_euler(self.matrix[0:3, 0:3])
        if isDegree:
            rx = math.degrees(rx)
            ry = math.degrees(ry)
            rz = math.degrees(rz)
        ret = [x, y, z, rx, ry, rz]
        if isQuaternion:
            qw, qx, qy, qz = euler2quat(x=rx, y=ry, z=rz)
            ret = [x, y, z, qx, qy, qz, qw]
        return np.array(ret)

    @staticmethod
    def from_parameters(x, y, z, euler_x, euler_y, euler_z, is_degree=False):
        ret = Transform()
        ret.set_translation(x, y, z)
        if is_degree:
            euler_x = math.radians(euler_x)
            euler_y = math.radians(euler_y)
            euler_z = math.radians(euler_z)
        ret.set_rotation(euler_x, euler_y, euler_z)
        return ret

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def rotation(self):
        ret = Transform()
        ret.matrix[0:3, 0:3] = self.matrix[0:3, 0:3]
        return ret

    @property
    def translation(self):
        ret = Transform()
        ret.matrix[0:3, 3] = self.matrix[0:3, 3]
        return ret

    def inverse(self):
        ret = Transform()
        ret.matrix[0:3, 0:3] = self.matrix[0:3, 0:3].transpose()
        ret.matrix[0:3, 3] = -ret.matrix[0:3, 0:3].dot(self.matrix[0:3, 3])
        return ret

    def transpose(self):
        ret = Transform()
        ret.matrix = self.matrix.T
        return ret

    def dot(self, points):
        shape = points.shape
        if shape[1] == 3:
            # to homogeneous (stack layer of one)
            ones = np.ones((shape[0], 1))
            homogeneous = np.hstack((points, ones)).T
        elif shape[1] == 4:
            homogeneous = points.T
        else:
            raise ValueError(
                "input array has to be of size 3 or in homogeneous coordinate, current size = " + str(shape))
        return self.matrix.dot(homogeneous).T[:, 0:3]

    def combine(self, transform, copy=False):
        ret_transform = self
        if not copy:
            self.matrix = self.matrix.dot(transform.matrix)
        else:
            new_matrix = self.matrix.dot(transform.matrix)
            ret_transform = Transform.from_matrix(new_matrix)
        return ret_transform

    def copy(self):
        ret = Transform()
        ret.matrix = self.matrix.copy()
        return ret

    def __getitem__(self, item):
        return self.matrix[item]

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __str__(self):
        params = self.to_parameters(isDegree=True)
        ret = ""
        ret += "x :" + str(params[0]) + ",\n"
        ret += "y :" + str(params[1]) + ",\n"
        ret += "z :" + str(params[2]) + ",\n"
        ret += "x :" + str(params[3]) + " degrees,\n"
        ret += "y :" + str(params[4]) + " degrees,\n"
        ret += "z :" + str(params[5]) + " degrees.\n"
        return ret

    def __repr__(self):
        return str(self.matrix)

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return np.isclose(self.matrix, other.matrix).all()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


def compute_delta_rotation_in_batches(batch_pose1, batch_pose2):
    """
    Given pose in batch, compute the matrix rotation between pose1, and pose2
    :param batch_pose1: Bx3x3
    :param batch_pose2: Bx3x3
    :return: Bx3X3 list matrix of rotation between batch1 and batch2
    """
    list_euler = np.zeros((len(batch_pose1), 3))
    for frame in range(len(batch_pose1)):
        pose1_i = batch_pose1[frame]  # rotation matrix
        pose2_i = batch_pose2[frame]
        delta_R_i = np.zeros((4, 4))
        delta_R_i[:3, :3] = pose2_i.dot(pose1_i.T)
        delta_R_i = Transform.from_matrix(delta_R_i)
        list_euler[frame, :] = delta_R_i.to_parameters(isDegree=False)[3:]
    return list_euler


def add_inplane(opencv_rotation, inplane):
    new_rotation = np.zeros_like(opencv_rotation)
    if len(new_rotation.shape) == 3:  # it's GT of shape (N, 3, 3)
        for i in range(len(opencv_rotation)):
            # convert from axis angles to matrix
            # add inplane rotation
            inplane_matrix = R.from_euler('z', -inplane[i], degrees=True).as_matrix()
            original_matrix_with_inplane = np.zeros((4, 4))
            original_matrix_with_inplane[3, 3] = 1
            new_rotation[i] = inplane_matrix.dot(opencv_rotation[i, :3, :3])
    else:  # it's deltaR of shape (N, 3) in axis angles representation
        # convert from axis angles to matrix
        original_matrix = Transform.from_parameters(0, 0, 0, opencv_rotation[0][0],
                                                    opencv_rotation[0][1], opencv_rotation[0][2]).matrix
        # add inplane rotation
        inplane_matrix = R.from_euler('z', -inplane[0], degrees=True).as_matrix()
        original_matrix_with_inplane = np.zeros((4, 4))
        original_matrix_with_inplane[3, 3] = 1
        original_matrix_with_inplane[:3, :3] = inplane_matrix.dot(original_matrix[:3, :3])
        original_matrix_with_inplane = Transform.from_matrix(original_matrix_with_inplane)
        new_rotation[0] = original_matrix_with_inplane.to_parameters(isDegree=False)[3:]
        # convert it back to axis-angle
    return new_rotation