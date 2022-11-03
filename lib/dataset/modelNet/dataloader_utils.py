import os
import numpy as np
from lib.poses import transform, utils


class SequenceProcessing:
    def __init__(self, root_path, dataset_name):
        self.root_path = root_path
        self.poses = np.load(os.path.join(root_path, "poses.npz"))["poses"]
        self.uv = utils.projection_perspective(self.poses[:, :3, 3], dataset_name=dataset_name)
        self.depth = self.poses[:, 2, 3].reshape(-1, 1)
        self.index_sequences = self.create_list_index_frame_in_sequences()

    def create_list_index_frame_in_sequences(self):
        """"
        Pairs 1: [0, 1]
        Pairs 1: [2, 3]
        Ex: [  0  16  32  48  64  80  96 112 128 144 160 176 192]
        Return:
            num_sequences X len_sequences
        """
        # create full index from start index and len_sequences
        index_frame_in_sequences = np.zeros((int(len(self.poses)/2), 2), dtype=np.int16)
        index_frame_in_sequences[:, 0] = np.arange(0, len(self.poses), 2)
        index_frame_in_sequences[:, 1] = index_frame_in_sequences[:, 0] + 1
        return index_frame_in_sequences

    def create_list_img_path(self):
        list_rgb_path = []
        for index_frame in range(2):
            list_index = self.index_sequences[:, index_frame]
            rgb_path = [os.path.join(self.root_path, "{:06d}.png".format(index)) for index in list_index]
            list_rgb_path.append(rgb_path)
        list_rgb_path = np.array(list_rgb_path).T
        return list_rgb_path

    def create_gt_tracking(self):
        list_delta_r = np.zeros((len(self.index_sequences), 1, 3))
        list_delta_uv = np.zeros((len(self.index_sequences), 1, 2))
        list_delta_d = np.zeros((len(self.index_sequences), 1, 1))

        list_rotation_first_frame = np.zeros((len(self.index_sequences), 3, 3))
        list_uv_first_frame = np.zeros((len(self.index_sequences), 2))
        list_d_first_frame = np.zeros((len(self.index_sequences), 1))

        list_gt_translations = np.zeros((len(self.index_sequences), 2, 3))
        list_gt_rotations = np.zeros((len(self.index_sequences), 2, 3, 3))

        for i in range(len(self.index_sequences)):
            list_delta_r[i] = transform.compute_delta_rotation_in_batches(
                self.poses[self.index_sequences[i, :-1], :3, :3],
                self.poses[self.index_sequences[i, 1:], :3, :3])
            # delta_uv w.r.t its previous frame
            list_rotation_first_frame[i] = self.poses[self.index_sequences[i, 0], :3, :3]
            list_uv_first_frame[i] = self.uv[self.index_sequences[i, 0], :]
            list_delta_uv[i] = self.uv[self.index_sequences[i, 1:], :] - self.uv[self.index_sequences[i, :-1], :]

            # delta_d w.r.t its previous frame
            list_d_first_frame[i] = self.depth[self.index_sequences[i, 0], :]
            list_delta_d[i] = self.depth[self.index_sequences[i, 1:], :] / self.depth[self.index_sequences[i, :-1],
                                                                           :] - 1
            # ground-truth translations
            list_gt_translations[i] = self.poses[self.index_sequences[i, :], :3, 3]
            list_gt_rotations[i] = self.poses[self.index_sequences[i, :], :3, :3]
        return dict(delta_rotation=list_delta_r, delta_uv=list_delta_uv, delta_depth=list_delta_d,
                    rotation_first_frame=list_rotation_first_frame, uv_first_frame=list_uv_first_frame,
                    depth_first_frame=list_d_first_frame, gt_rotations=list_gt_rotations,
                    gt_translations=list_gt_translations)
