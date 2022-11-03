import math
import numpy as np
import torch


def cum_product(tensor_3d):
    tensor_3d = torch.log(tensor_3d)
    tensor_3d = torch.cumsum(tensor_3d, axis=1)  # BxLx1
    tensor_3d = torch.exp(tensor_3d)
    return tensor_3d


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                   2], norm_quat[:,
                                                       3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
        dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    # axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    # rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def rotationMatrixToEulerAngles(batch_matrix_rotation):
    """
    :param batch_matrix_rotation: Bx3x3 list matrix of rotation
    :return: euler angles corresponding to each matrix
    """
    batch_size = len(batch_matrix_rotation)
    error_euler = np.zeros((batch_size, 3))
    for i in range(batch_size):
        R = batch_matrix_rotation[i]
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            z = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            x = math.atan2(R[1, 0], R[0, 0])
        else:
            z = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            x = 0
        error_euler[i] = [x, y, z]
    return error_euler