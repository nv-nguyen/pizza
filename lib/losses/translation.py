import torch
import numpy as np
from lib.losses.utils import cum_product
from lib.poses.utils import intrinsic_modelNet_inverse

intrinsic_modelNet_inverse_tensor = torch.from_numpy(np.ascontiguousarray(intrinsic_modelNet_inverse)).float()
name_translation_metrics = ["translation_err", "X_err", "Y_err", "Z_err"]


def measure_translation(delta_uv_pred, delta_d_pred,
                        uv_first_frame, d_first_frame,
                        alpha_resize, gt_delta_uv, gt_delta_d,
                        cumulative, loss_function, dataset_name,
                        in_mm=True, with_cumulative_err=True, img_crop_size=224):
    """
    :param gt_delta_uv:
    :param delta_uv_pred: Bx(L-1)x2, prediction of delta_U, delta_V of frame t w.r.t frame t+1
    :param delta_d_pred: Bx(L-1)x1, prediction of S of frame t w.r.t frame t+1
    :param uv_first_frame: Bx1, (GT data) U,V of first frame from given pose
    :param d_first_frame: Bx1, (GT data) Z of first frame from given pose
    :param alpha_resize: Bx1, resizing factor of first frame from given pose
    :param loss_function: function to calculate loss, supposed to be the L2 with reduction="sum"
    :param dataset_name: ModelNet or Laval
    :param img_crop_size: size of cropped image
    :param with_cumulative_err: whether get also with_cumulative_err error of the sequences
    :param in_mm: in mm
    :return:
    """
    # get intrinsic matrix given the name of dataset
    if dataset_name == "modelNet":
        intrinsic_inverse = intrinsic_modelNet_inverse_tensor.cuda(delta_uv_pred.get_device())
    batch_size, len_sequences_minus = delta_uv_pred.size(0), delta_uv_pred.size(1)

    # starting to compute the predicted translation
    if alpha_resize.size(1) == 1:  #
        single_frame = True
        alpha_resize = alpha_resize.unsqueeze(1)  # Bx1x1
        if cumulative:
            delta_UV_wrt_first_frame = torch.cumsum(delta_uv_pred / alpha_resize * (img_crop_size / 2), axis=1)
        else:
            delta_UV_wrt_first_frame = delta_uv_pred / alpha_resize * (img_crop_size / 2)
    else:
        single_frame = False
        if cumulative:
            delta_UV_wrt_first_frame = torch.cumsum(delta_uv_pred / alpha_resize * (img_crop_size / 2), axis=1)
        else:
            delta_UV_wrt_first_frame = delta_uv_pred / alpha_resize * (img_crop_size / 2)

    uv = uv_first_frame.unsqueeze(1).repeat(1, len_sequences_minus, 1) + delta_UV_wrt_first_frame
    uv_homogeneous = torch.cat((uv, torch.ones(batch_size, len_sequences_minus, 1).float().cuda()), axis=2)
    # coord is [X,Y,1]
    coord = torch.bmm(intrinsic_inverse.unsqueeze(0).repeat(batch_size, 1, 1),
                      uv_homogeneous.permute(0, 2, 1)).permute(0, 2, 1)
    # now, get Z value
    if single_frame:
        delta_d_pred = delta_d_pred / alpha_resize + 1
    else:
        delta_d_pred = delta_d_pred / alpha_resize[:, :, 0].unsqueeze(-1) + 1
    if cumulative:
        d = cum_product(delta_d_pred) * d_first_frame.unsqueeze(1).repeat(1, len_sequences_minus, 1)
    else:
        d = delta_d_pred * d_first_frame.unsqueeze(1).repeat(1, len_sequences_minus, 1)
    translation_pred = coord * d
    translation_pred = translation_pred.float()

    # starting to compute the gt translation
    if alpha_resize.size(1) == 1:  #
        gt_delta_UV_wrt_first_frame = gt_delta_uv / alpha_resize * (img_crop_size / 2)
    else:
        single_frame = False
        gt_delta_UV_wrt_first_frame = gt_delta_uv / alpha_resize * (img_crop_size / 2)

    gt_uv = uv_first_frame.unsqueeze(1).repeat(1, len_sequences_minus, 1) + gt_delta_UV_wrt_first_frame
    gt_uv_homogeneous = torch.cat((gt_uv, torch.ones(batch_size, len_sequences_minus, 1).float().cuda()), axis=2)
    # coord is [X,Y,1]
    gt_coord = torch.bmm(intrinsic_inverse.unsqueeze(0).repeat(batch_size, 1, 1),
                         gt_uv_homogeneous.permute(0, 2, 1)).permute(0, 2, 1)
    # now, get Z value
    if single_frame:
        gt_delta_d = gt_delta_d / alpha_resize + 1
    else:
        gt_delta_d = gt_delta_d / alpha_resize[:, :, 0].unsqueeze(-1) + 1
    gt_d = gt_delta_d * d_first_frame.unsqueeze(1).repeat(1, len_sequences_minus, 1)
    translations_gt_center_crop = gt_coord * gt_d
    translations_gt_center_crop = translations_gt_center_crop.float()

    # we do not take into account the first frame, this is why translations_gt->translations_gt[:, 1:, :]
    translation_err = loss_function(translation_pred, translations_gt_center_crop)
    X_err = loss_function(translation_pred[:, :, 0], translations_gt_center_crop[:, :, 0])
    Y_err = loss_function(translation_pred[:, :, 1], translations_gt_center_crop[:, :, 1])
    Z_err = loss_function(translation_pred[:, :, 2], translations_gt_center_crop[:, :, 2])

    # as the loss_function is defined with reduction="sum" and **2, we have to rescale it to get the correct mean
    N_elements = batch_size * len_sequences_minus  # num_elements in loss function
    translation_err = torch.sqrt(translation_err / N_elements)
    X_err = torch.sqrt(X_err / N_elements)
    Y_err = torch.sqrt(Y_err / N_elements)
    Z_err = torch.sqrt(Z_err / N_elements)

    if with_cumulative_err:
        cumulative_err = torch.abs(translation_pred - translations_gt_center_crop)  # BxLx3
        cumulative_err = torch.norm(cumulative_err, dim=2)  # BxL
        cumulative_err = torch.mean(cumulative_err, dim=0)  # L
        cumulative_err = torch.mean(cumulative_err, dim=0)
    else:
        cumulative_err = None

    if in_mm:
        translation_err = translation_err * (10 ** 3)
        X_err = X_err * (10 ** 3)
        Y_err = Y_err * (10 ** 3)
        Z_err = Z_err * (10 ** 3)
        if with_cumulative_err:
            cumulative_err = cumulative_err * (10 ** 3)

    return dict(translation_pred=translation_pred, translations_gt_center_crop=translations_gt_center_crop,
                translation_err=translation_err, cumulative_err=cumulative_err,
                X_err=X_err, Y_err=Y_err, Z_err=Z_err, uv=uv, d=d)
