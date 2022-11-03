import os, time
import torch
import torch.nn as nn
import json
from lib.utils.metrics import AverageValueMeter
from lib.losses.rotation import measure_rotation, name_rotation_metrics
from lib.losses.translation import measure_translation, name_translation_metrics
from bop_toolkit_lib.metrics import evaluate_6d_metrics

names = ["cad_name", "img_path", "T_pred", "T_gt", "R_pred", "R_gt"]
MSELoss = nn.MSELoss()


def init_prediction():
    all_predictions_and_gt = {}
    for name in names:
        all_predictions_and_gt[name] = []
    return all_predictions_and_gt


def collect_prediction(all_predictions_and_gt, cad_name, T_pred, T_gt, R_pred, R_gt, img_path):
    all_data = [cad_name, img_path, T_pred, T_gt, R_pred, R_gt]
    for idx, data in enumerate(all_data):
        if idx >= 2:
            data_numpy = data.cpu().detach().numpy()
        else:
            data_numpy = data
        all_predictions_and_gt[names[idx]].extend(data_numpy)


def save_prediction(save_path, all_predictions_and_gt):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    all_predictions_and_gt_list = {}
    for name in names:
        data = all_predictions_and_gt[name]
        if name not in ["cad_name", "img_path"]:
            data_list = []
            for i in range(len(data)):
                data_list.append(data[i].tolist())
            all_predictions_and_gt_list[name] = data_list
        else:
            all_predictions_and_gt_list[name] = data
    with open(save_path, 'w') as f:
        json.dump(all_predictions_and_gt_list, f, indent=4)
    print("Saving to {} done!".format(save_path))


def test(test_data, model, dataset_name, cat, data_path, save_path, predict_translation, epoch, logger,
                  log_interval, tb_logger, is_master):
    start_time = time.time()
    meter_testing_loss = AverageValueMeter()
    meter_rotations = {name_rotation_metrics[i]: AverageValueMeter() for i in range(len(name_rotation_metrics))}
    if predict_translation:
        meter_UV = AverageValueMeter()
        meter_Z = AverageValueMeter()
        meter_translation = {name_translation_metrics[i]: AverageValueMeter()
                             for i in range(len(name_translation_metrics))}
        monitoring_text = 'Testing-Epoch-{} -- Iter [{}/{}] loss: {:.2f} (Avg: {:.2f}), ' \
                          'Rotation: {:.2f} (X={:.2f}, Y={:.2f}, Z={:.2f}) ' \
                          'Translation: UV={:.2f}, Z={:.2f}, T={:.2f} (X={:.2f}, Y={:.2f}, Z={:.2f})'
    else:
        monitoring_text = 'Testing-Epoch-{} -- Iter [{}/{}] loss: {:.2f} (Avg: {:.2f}), ' \
                          'Rotation: {:.2f} (X={:.2f}, Y={:.2f}, Z={:.2f})'
    timing_text = "Testing time for epoch {}: {:.02f} minutes"

    all_predictions_and_gt = init_prediction()

    model.eval()
    with torch.no_grad():
        test_size, test_loader = len(test_data), iter(test_data)
        for i in range(test_size):
            batch = test_loader.next()
            seq_img = batch['seq_img'].cuda()
            gt_delta_rotation = batch['delta_rotation'].cuda()
            gt_rotations = batch['gt_rotations'].cuda()
            rotation_first_frame = batch['rotation_first_frame'].cuda()
            if not predict_translation:
                R_pred = model(seq_img)
            else:
                R_pred, UV_pred, D_pred = model(seq_img)
                ratio = batch['ratio'].cuda()
                cad_name = batch['cad_name']
                gt_delta_uv = batch['delta_uv'].cuda()
                gt_delta_depth = batch['delta_depth'].cuda()
                uv_first_frame = batch['uv_first_frame'].cuda()
                depth_first_frame = batch['depth_first_frame'].cuda()
                gt_translations = batch['gt_translations'].cuda()
                img_path = batch['img_path']
                # calculate translation error
                T_metrics = measure_translation(delta_uv_pred=UV_pred, delta_d_pred=D_pred,
                                                uv_first_frame=uv_first_frame, d_first_frame=depth_first_frame,
                                                alpha_resize=ratio, cumulative=True,
                                                gt_delta_uv=gt_delta_uv, gt_delta_d=gt_delta_depth,
                                                loss_function=nn.MSELoss(reduction="sum").cuda(
                                                    depth_first_frame.get_device()),
                                                dataset_name=dataset_name)
                for name in name_translation_metrics:
                    meter_translation[name].update(T_metrics[name].item())
            # calculate rotation error
            R_metrics = measure_rotation(axis_angles_pred=R_pred, axis_angles_gt=gt_delta_rotation,
                                         rot_first_frame=rotation_first_frame, rots_gt=gt_rotations,
                                         cumulative=True)

            collect_prediction(all_predictions_and_gt=all_predictions_and_gt,
                               cad_name=cad_name,
                               T_pred=T_metrics["translation_pred"],
                               T_gt=gt_translations[:, 1:, ],
                               R_pred=R_metrics['rotation_pred'],
                               R_gt=gt_rotations[:, 1:, ],
                               img_path=img_path)

            for name in name_rotation_metrics:
                meter_rotations[name].update(R_metrics[name].item())

            if predict_translation:
                loss_Z = MSELoss(D_pred, gt_delta_depth)
                loss_UV = MSELoss(UV_pred, gt_delta_uv)
                loss = loss_Z + loss_UV + R_metrics["geodesic_err"]
                meter_UV.update(loss_UV.item())
                meter_Z.update(loss_Z.item())
            else:
                loss = R_metrics["geodesic_err"]
            meter_testing_loss.update(loss.item())

            if i % log_interval == 0 and is_master:
                torch.cuda.synchronize()
                # Add log into log file
                if predict_translation:
                    stat = monitoring_text.format(epoch, i, test_size, meter_testing_loss.val,
                                                  meter_testing_loss.avg,
                                                  meter_rotations["geodesic_err"].val,
                                                  meter_rotations["X_err"].val,
                                                  meter_rotations["Y_err"].val, meter_rotations["Z_err"].val,
                                                  meter_UV.val, meter_Z.val,
                                                  meter_translation["translation_err"].val,
                                                  meter_translation["X_err"].val,
                                                  meter_translation["Y_err"].val, meter_translation["Z_err"].val)
                else:
                    stat = monitoring_text.format(epoch, i, test_size, meter_testing_loss.val,
                                                  meter_testing_loss.avg,
                                                  meter_rotations["geodesic_err"].val, meter_rotations["X_err"].val,
                                                  meter_rotations["Y_err"].val, meter_rotations["Z_err"].val)
                logger.info(stat)
    save_prediction(save_path=save_path, all_predictions_and_gt=all_predictions_and_gt)
    logger.info(timing_text.format(epoch, (time.time() - start_time) / 60))
    score = evaluate_6d_metrics(all_predictions_and_gt=all_predictions_and_gt, data_path=data_path,
                                metric="5deg_5cm")
    if is_master:
        if predict_translation:
            tb_logger.add_scalar_dict_list('train', [{'5deg_5cm, {}'.format(cat): score["5deg_5cm"],
                                                      'test_loss: {}'.format(cat): meter_testing_loss.avg,
                                                      'test_UV_loss: {}'.format(cat): meter_UV.avg,
                                                      'test_Z_loss': meter_Z.avg,
                                                      'testing geodesic_err: {}'.format(cat): meter_rotations[
                                                          "geodesic_err"].avg,
                                                      'testing translation_err: {}'.format(cat): meter_translation[
                                                          "translation_err"].avg}], epoch)
    return score["5deg_5cm"]
