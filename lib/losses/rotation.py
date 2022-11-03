import torch
import numpy as np
from lib.losses.utils import batch_rodrigues, rotationMatrixToEulerAngles
name_rotation_metrics = ["geodesic_err", "X_err", "Y_err", "Z_err"]


def geodesic_distance_from_two_batches(batch1, batch2, with_euler_error=True):
    """

    :param batch1: Bx3x3 B matrix of rotation
    :param batch2: Bx3x3 B matrix of rotation
    :param with_euler_error: whether output euler error for each axis
    :return: B theta angles of the matrix rotation from batch1 to batch2
    """
    batch = batch1.shape[0]
    m = torch.bmm(batch1, batch2.transpose(1, 2))  # batch*3*3
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)
    theta = torch.acos(cos)

    if with_euler_error:
        m = m.cpu().detach().numpy()
        euler_angles = rotationMatrixToEulerAngles(m)
        error_euler = np.mean(np.abs(euler_angles), axis=0)
    else:
        error_euler = None
    return theta, error_euler


def measure_rotation(axis_angles_pred, axis_angles_gt, rot_first_frame, rots_gt, cumulative,
                     with_euler_error=True, in_degree=True, with_cumulative_err=True):
    """

    :param axis_angles_pred:
    :param axis_angles_gt:
    :param rot_first_frame:
    :param rots_gt:
    :param cumulative: the delta rotation is calculated cumulatively, delta(1,2), delta(2,3)
    :param with_euler_error:
    :param in_degree:
    :param with_cumulative_err:
    :return:
    """
    batch_size, len_sequences_minus = axis_angles_pred.size(0), axis_angles_pred.size(1)
    # convert axis-angles to rotation matrix
    delta_r_mat_pred = batch_rodrigues(axis_angles_pred.view(-1, 3)).view(batch_size, len_sequences_minus, 3, 3)
    delta_r_mat_gt = batch_rodrigues(axis_angles_gt.view(-1, 3)).view(batch_size, len_sequences_minus, 3, 3)
    # get the prediction from previous frame and the current delta
    previous_rotation = rot_first_frame
    pred_rotation = torch.zeros(batch_size, len_sequences_minus, 3, 3).float().cuda()
    for i in range(len_sequences_minus):
        pred_rotation[:, i, :, :] = torch.bmm(delta_r_mat_pred[:, i, :, :], previous_rotation)
        if cumulative:
            previous_rotation = pred_rotation[:, i, :, :]
    # get the tracking geodesic distance between prediction and ground-truth
    geodesic_err, error_euler = geodesic_distance_from_two_batches(delta_r_mat_pred.reshape(-1, 3, 3),
                                                                   delta_r_mat_gt.reshape(-1, 3, 3),
                                                                   with_euler_error=with_euler_error)
    # get the tracking geodesic distance between prediction and ground-truth
    tracking_geodesic_err, tracking_error_euler = geodesic_distance_from_two_batches(pred_rotation.reshape(-1, 3, 3),
                                                                                     rots_gt[:, 1:, :, :].reshape(-1, 3, 3),
                                                                                     with_euler_error=with_euler_error)
    if with_cumulative_err:
        cumulative_err = tracking_geodesic_err.reshape(batch_size, len_sequences_minus)  # reshape from (Bx(L-1)) to
        # Bx(L-1)
        cumulative_err = torch.mean(cumulative_err, dim=0)  # get error w.r.t temporal frame -> (L-1)
        cumulative_err = torch.cumsum(cumulative_err, dim=0)  # get cumulative error
    else:
        cumulative_err = None

    geodesic_err = geodesic_err.mean()
    tracking_geodesic_err = tracking_geodesic_err.mean()
    if in_degree:
        geodesic_err = torch.rad2deg(geodesic_err)
        error_euler = np.rad2deg(error_euler)
        tracking_geodesic_err = torch.rad2deg(tracking_geodesic_err)
        tracking_error_euler = np.rad2deg(tracking_error_euler)
        if with_cumulative_err:
            cumulative_err = torch.rad2deg(cumulative_err)
    return dict(geodesic_err=geodesic_err, tracking_geodesic_err=tracking_geodesic_err,
                cumulative_err=cumulative_err,
                X_err=error_euler[2], Y_err=error_euler[1], Z_err=error_euler[0],
                X_tracking_err=tracking_error_euler[2], Y_tracking_err=tracking_error_euler[1],
                Z_tracking_err=tracking_error_euler[0],
                rotation_pred=pred_rotation)
