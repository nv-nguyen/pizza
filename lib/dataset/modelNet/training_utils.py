import time
import torch
import torch.nn as nn

from lib.utils.metrics import AverageValueMeter
from lib.losses.rotation import measure_rotation, name_rotation_metrics
from lib.losses.translation import measure_translation, name_translation_metrics

MSELoss = nn.MSELoss()


def train(train_data, model, dataset_name, predict_translation, optimizer, warm_up_config, epoch, logger, tb_logger, log_interval,
          is_master):
    start_time = time.time()
    meter_training_loss = AverageValueMeter()
    meter_rotations = {name_rotation_metrics[i]: AverageValueMeter() for i in range(len(name_rotation_metrics))}
    if predict_translation:
        meter_UV = AverageValueMeter()
        meter_Z = AverageValueMeter()
        meter_translation = {name_translation_metrics[i]: AverageValueMeter()
                             for i in range(len(name_translation_metrics))}
        monitoring_text = 'Training-Epoch-{} -- Iter [{}/{}] loss: {:.2f} (Avg: {:.2f}), ' \
                          'Rotation: {:.2f} (X={:.2f}, Y={:.2f}, Z={:.2f}) ' \
                          'Translation: UV={:.2f}, Z={:.2f}, T={:.2f} (X={:.2f}, Y={:.2f}, Z={:.2f})'
    else:
        monitoring_text = 'Epoch-{} -- Iter [{}/{}] loss: {:.2f} (Avg: {:.2f}), ' \
                          'Rotation: {:.2f} (X={:.2f}, Y={:.2f}, Z={:.2f})'
    timing_text = "Training time for epoch {}: {:.02f} minutes"

    model.train()
    train_size, train_loader = len(train_data), iter(train_data)
    print("Train_size", train_size)
    with torch.autograd.set_detect_anomaly(True):
        for i in range(train_size):
            # update learning rate with warm up
            if warm_up_config is not None:
                [nb_iter_warm_up, lr] = warm_up_config
                nb_iter = epoch * train_size + i
                if nb_iter <= nb_iter_warm_up:
                    lrUpdate = nb_iter / float(nb_iter_warm_up) * lr
                    for g in optimizer.param_groups:
                        g['lr'] = lrUpdate

                batch = train_loader.next()
                seq_img = batch['seq_img'].cuda()
                gt_delta_rotation = batch['delta_rotation'].cuda()
                gt_rotations = batch['gt_rotations'].cuda()
                rotation_first_frame = batch['rotation_first_frame'].cuda()
                if not predict_translation:
                    R_pred = model(seq_img)
                else:
                    R_pred, UV_pred, D_pred = model(seq_img)
                    ratio = batch['ratio'].cuda()
                    uv_first_frame = batch['uv_first_frame'].cuda()
                    gt_delta_uv = batch['delta_uv'].cuda()
                    gt_delta_depth = batch['delta_depth'].cuda()
                    depth_first_frame = batch['depth_first_frame'].cuda()
                    gt_translations = batch['gt_translations'].cuda()
                    # calculate translation error
                    T_metrics = measure_translation(delta_uv_pred=UV_pred, delta_d_pred=D_pred,
                                                    uv_first_frame=uv_first_frame, d_first_frame=depth_first_frame,
                                                    gt_delta_uv=gt_delta_uv, gt_delta_d=gt_delta_depth,
                                                    alpha_resize=ratio, cumulative=True,
                                                    loss_function=nn.MSELoss(reduction="sum").cuda(
                                                        depth_first_frame.get_device()),
                                                    dataset_name=dataset_name)
                    for name in name_translation_metrics:
                        meter_translation[name].update(T_metrics[name].item())
                # calculate rotation error
                R_metrics = measure_rotation(axis_angles_pred=R_pred, axis_angles_gt=gt_delta_rotation,
                                             rot_first_frame=rotation_first_frame, rots_gt=gt_rotations,
                                             cumulative=True)
                for name in name_rotation_metrics:
                    meter_rotations[name].update(R_metrics[name].item())

                if predict_translation:
                    loss_Z = MSELoss(D_pred, gt_delta_depth) * (10 ** 3)  # to mm
                    loss_UV = MSELoss(UV_pred, gt_delta_uv) * (10 ** 3)  # to mm
                    loss = loss_Z + loss_UV + R_metrics["geodesic_err"]
                else:
                    loss = R_metrics["geodesic_err"]
                meter_training_loss.update(loss.item())

                optimizer.zero_grad()
                loss.backward()
                # Call step of optimizer to update model params
                optimizer.step()

                if i % log_interval == 0 and is_master:
                    torch.cuda.synchronize()
                    # Add log into log file
                    if predict_translation:
                        stat = monitoring_text.format(epoch, i, train_size, meter_training_loss.val,
                                                      meter_training_loss.avg,
                                                      meter_rotations["geodesic_err"].val,
                                                      meter_rotations["X_err"].val,
                                                      meter_rotations["Y_err"].val, meter_rotations["Z_err"].val,
                                                      meter_UV.val, meter_Z.val,
                                                      meter_translation["translation_err"].val,
                                                      meter_translation["X_err"].val,
                                                      meter_translation["Y_err"].val, meter_translation["Z_err"].val)
                    else:
                        stat = monitoring_text.format(epoch, i, train_size, meter_training_loss.val,
                                                      meter_training_loss.avg,
                                                      meter_rotations["geodesic_err"].val, meter_rotations["X_err"].val,
                                                      meter_rotations["Y_err"].val, meter_rotations["Z_err"].val)
                    logger.info(stat)
    logger.info(timing_text.format(epoch, (time.time() - start_time) / 60))
    if is_master:
        if predict_translation:
            tb_logger.add_scalar_dict_list('train', [{'train_loss': meter_training_loss.avg,
                                                    'geodesic_err': meter_rotations["geodesic_err"].avg,
                                                    'translation_err': meter_translation["translation_err"].avg}], epoch)
    return meter_training_loss.avg
