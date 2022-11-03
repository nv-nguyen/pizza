import os
import json
import numpy as np
from tqdm import tqdm
from lib.poses.utils import geodesic_numpy, get_intrinsic_matrix
import trimesh
import random
import glumpy

synthetic_names = ["cad_name", "img_path", "T_pred", "T_gt", "R_pred", "R_gt"]
real_names = ["T_pred_center_crop", "T_gt_center_crop", "T_gt", "R_pred", "R_gt",
              "index_frames", "rotation_first_frame"]


def init_prediction():  # init dictionary to save prediction for synthetic data
    all_predictions_and_gt = {}
    for name in synthetic_names:
        all_predictions_and_gt[name] = []
    return all_predictions_and_gt


def collect_prediction(all_predictions_and_gt, cad_name, T_pred, T_gt, R_pred, R_gt, img_path):
    all_data = [cad_name, img_path, T_pred, T_gt, R_pred, R_gt]
    for idx, data in enumerate(all_data):
        if idx >= 2:
            data_numpy = data.cpu().detach().numpy()
        else:
            data_numpy = data
        all_predictions_and_gt[synthetic_names[idx]].extend(data_numpy)


def save_prediction(save_path, all_predictions_and_gt):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    all_predictions_and_gt_list = {}
    for name in synthetic_names:
        print(name)
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


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        result = trimesh.util.concatenate([trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                                           for m in scene_or_mesh.geometry.values()])
    else:
        result = scene_or_mesh
    return result


def load_mesh(path, rescale, num_points):
    mesh = as_mesh(trimesh.load(path))
    AABB = mesh.bounds
    center = np.mean(AABB, axis=0)
    mesh.vertices -= center
    if rescale:  # normalize object in -1 and 1 as glumpy
        AABB = mesh.bounds
        scale = np.max(AABB[1, :] - AABB[0, :], axis=0)
        mesh.vertices /= (scale * 0.5)
    np.random.seed(2022)
    random.seed(2022)
    return mesh.sample(num_points)


def evaluate_6d_metrics(all_predictions_and_gt, metric,
                        data_path, dataset_name='modelNet', percentage=1):
    assert metric in ["5deg_5cm", "add", "proj", "all"], print("Metric to evaluate is not correct!!!")
    num_images = int(len(all_predictions_and_gt["T_pred"]) * percentage)
    output = dict()
    if metric in ['all', 'add', 'proj']:
        obj_models = dict()
        for i in tqdm(range(num_images)):
            obj_path = all_predictions_and_gt['cad_name'][i]
            if obj_path.split('/')[-1] not in obj_models:
                obj_models[obj_path.split('/')[-1]] = dict()
                obj, _ = glumpy.data.objload(data_path + obj_path + '.obj', rescale=True)
                obj = np.array(obj["position"] / 10.0)
                obj_diameter = np.linalg.norm(obj.max(0) - obj.min(0))
                obj_models[obj_path.split('/')[-1]]['diameter'] = obj_diameter
                # some objects in ModelNet have very few points, we need to sample more points
                obj = load_mesh(data_path + obj_path + '.obj', rescale=True, num_points=10000)
                obj = np.array(obj / 10.0)
                obj_models[obj_path.split('/')[-1]]['pts'] = obj

    if metric in ["5deg_5cm", "all"]:
        output['5deg_5cm'] = {}
        for i in range(num_images):
            R_pred = np.asarray(all_predictions_and_gt["R_pred"][i]).reshape(3, 3)
            R_gt = np.asarray(all_predictions_and_gt["R_gt"][i]).reshape(3, 3)
            err_R = geodesic_numpy(R1=R_pred, R2=R_gt)

            T_pred = np.asarray(all_predictions_and_gt["T_pred"][i]).reshape(3)
            T_gt = np.asarray(all_predictions_and_gt["T_gt"][i]).reshape(3)
            err_T = np.linalg.norm(T_gt - T_pred)
            if all_predictions_and_gt["cad_name"][i] not in output['5deg_5cm']:
                output['5deg_5cm'][all_predictions_and_gt["cad_name"][i]] = []
            score = np.logical_and(err_T * 100 <= 5, err_R <= 5) * 1
            output['5deg_5cm'][all_predictions_and_gt["cad_name"][i]].append(score)
        num_cad = len(output['5deg_5cm'])
        scores_5deg_5cm = np.zeros(num_cad)
        list_cad_name = []
        for idx_cad, cad in enumerate(output['5deg_5cm'].keys()):
            scores_5deg_5cm[idx_cad] = np.mean(output['5deg_5cm'][cad])
            list_cad_name.append(cad)
        list_cad_name = np.asarray(list_cad_name)
        output['5deg_5cm'] = np.mean(scores_5deg_5cm)

    if metric in ["add", "all"]:
        output['add'] = {}
        for i in range(num_images):
            obj_info = obj_models[all_predictions_and_gt["cad_name"][i].split('/')[-1]]
            R_pred = np.asarray(all_predictions_and_gt["R_pred"][i]).reshape(3, 3)
            R_gt = np.asarray(all_predictions_and_gt["R_gt"][i]).reshape(3, 3)
            T_pred = np.asarray(all_predictions_and_gt["T_pred"][i]).reshape(3, 1)
            T_gt = np.asarray(all_predictions_and_gt["T_gt"][i]).reshape(3, 1)

            pred_obj = (R_pred.dot(obj_info['pts'].T) + T_pred).T
            gt_obj = (R_gt.dot(obj_info['pts'].T) + T_gt).T
            error = np.linalg.norm(pred_obj - gt_obj, axis=1).mean()
            threshold_010 = 0.10 * obj_info['diameter']
            if all_predictions_and_gt["cad_name"][i] not in output['add']:
                output['add'][all_predictions_and_gt["cad_name"][i]] = []
            score = (threshold_010 >= error).astype(np.float32)
            output['add'][all_predictions_and_gt["cad_name"][i]].append(score)

        num_cad = len(output['add'])
        scores_add = np.zeros(num_cad)
        for idx_cad, cad in enumerate(output['add'].keys()):
            scores_add[idx_cad] = np.mean(output['add'][cad])
        output['add'] = np.mean(scores_add)

    if metric in ["proj", "all"]:
        output['proj'] = {}
        K = get_intrinsic_matrix(dataset_name)
        for i in range(num_images):
            obj_info = obj_models[all_predictions_and_gt["cad_name"][i].split('/')[-1]]
            R_pred = np.asarray(all_predictions_and_gt["R_pred"][i]).reshape(3, 3)
            R_gt = np.asarray(all_predictions_and_gt["R_gt"][i]).reshape(3, 3)
            T_pred = np.asarray(all_predictions_and_gt["T_pred"][i]).reshape(3, 1)
            T_gt = np.asarray(all_predictions_and_gt["T_gt"][i]).reshape(3, 1)

            pred_obj = K.dot(R_pred.dot(obj_info['pts'].T) + T_pred)
            n = obj_info['pts'].shape[0]
            pred_obj_new = np.zeros((n, 2))
            pred_obj_new[:, 0] = pred_obj[0, :] / pred_obj[2, :]
            pred_obj_new[:, 1] = pred_obj[1, :] / pred_obj[2, :]

            gt_obj = K.dot(R_gt.dot(obj_info['pts'].T) + T_gt)
            gt_obj_new = np.zeros((n, 2))
            gt_obj_new[:, 0] = gt_obj[0, :] / gt_obj[2, :]
            gt_obj_new[:, 1] = gt_obj[1, :] / gt_obj[2, :]

            error = np.linalg.norm(pred_obj_new - gt_obj_new, axis=1).mean()
            if all_predictions_and_gt["cad_name"][i] not in output['proj']:
                output['proj'][all_predictions_and_gt["cad_name"][i]] = []
            score = (5 >= error).astype(np.float32)
            output['proj'][all_predictions_and_gt["cad_name"][i]].append(score)

        num_cad = len(output['proj'])
        scores_add = np.zeros(num_cad)
        for idx_cad, cad in enumerate(output['proj'].keys()):
            scores_add[idx_cad] = np.mean(output['proj'][cad])
        output['proj'] = np.mean(scores_add)
    return output