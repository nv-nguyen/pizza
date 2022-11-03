import os, sys
import time
import numpy as np
import torch.utils.data as data
import torch
from PIL import Image

np.random.seed(2022)
from lib.utils.logger import print_and_log_info
from lib.dataloader import image_utils, utils, augmentation
from lib.dataset.modelNet import dataloader_utils
names = ["cad_name", "img_full_path", "delta_rotation", "delta_uv", "delta_depth",
         "rotation_first_frame", "uv_first_frame", "depth_first_frame",
         "gt_rotations", "gt_translations"]


class ModelNet(data.Dataset):
    def __init__(self, root_dir, split, config_training, logger, is_master):
        self.is_master = is_master
        self.logger = logger
        self.root_dir = root_dir
        self.split = split
        self.use_augmentation = config_training.use_augmentation
        self.image_size = config_training.image_size
        self.save_path = config_training.save_path
        self.debug_mode = config_training.debug_mode  # debugging mode (to visualize samples and loader quickly)

        self.sequence_data = self.get_data_from_split_name()
        if self.is_master:
            print_and_log_info(self.logger,
                               "Size of dataloader: {}".format(sys.getsizeof(self.sequence_data) / (10 ** 9)))
            print_and_log_info(self.logger, "Len of dataset :{}".format(self.__len__()))
            self.save_random_sequences()

    def __len__(self):
        return len(self.sequence_data["img_full_path"])

    def get_data_from_split_name(self):
        start_time = time.time()
        list_files = os.path.join(self.root_dir, self.split + ".txt")
        with open(list_files, 'r') as f:
            list_id_model = [x.strip() for x in f.readlines()]
        if self.debug_mode:
            list_id_model = list_id_model[:500]

        sequence_data = {names[i]: [] for i in range(len(names))}
        for id_model in list_id_model:
            path_trajectory = os.path.join(self.root_dir, id_model)
            sequence_obj = dataloader_utils.SequenceProcessing(root_path=path_trajectory, dataset_name="modelNet")
            gt_sequence_obj = sequence_obj.create_gt_tracking()
            sequence_data["img_full_path"].extend(sequence_obj.create_list_img_path())
            sequence_data["cad_name"].extend([id_model for _ in range(len(sequence_obj.create_list_img_path()))])
            for name in names[2:]:
                sequence_data[name].extend(gt_sequence_obj[name])
        if str(self.split).endswith("train"):
            print("Shuffling data before training...")
            sequence_data = utils.shuffle_dictionary(sequence_data)
        if self.is_master:
            print_and_log_info(self.logger, "Loading dataLoader takes {} seconds".format(time.time() - start_time))
        return sequence_data

    def _fetch_sequence(self, img_path, save_path=None):
        sequence_img, list_bbox = [], []
        for i in range(2):
            img = image_utils.open_image(img_path[i])
            sequence_img.append(img)
            list_bbox.append(np.asarray(img.getbbox()))
        # take max bbox of two images
        bbox_sequence = np.zeros(4)
        bbox_sequence[0] = np.min([list_bbox[0][0], list_bbox[1][0]])
        bbox_sequence[1] = np.min([list_bbox[0][1], list_bbox[1][1]])
        bbox_sequence[2] = np.max([list_bbox[0][2], list_bbox[1][2]])
        bbox_sequence[3] = np.max([list_bbox[0][3], list_bbox[1][3]])

        bbox_size = np.max([bbox_sequence[2] - bbox_sequence[0], bbox_sequence[3] - bbox_sequence[1]])
        max_size_with_margin = bbox_size * 1.3  # margin = 0.2 x max_dim
        margin = bbox_size * 0.15
        bbox_sequence = bbox_sequence + np.array([-margin, -margin, margin, margin])
        bbox_sequence_square = image_utils.make_bbox_square(bbox_sequence, max_size_with_margin)
        ratio = self.image_size / max_size_with_margin  # keep this value to predict translation later
        for i in range(2):
            cropped_img = sequence_img[i].crop(bbox_sequence_square)
            cropped_resized_img = cropped_img.resize((self.image_size, self.image_size), Image.ANTIALIAS)
            sequence_img[i] = cropped_resized_img
        if "train" in self.split and self.use_augmentation:
            sequence_img = augmentation.apply_data_augmentation(2, sequence_img)
        if save_path is None:
            seq_img = np.zeros((2, 3, self.image_size, self.image_size))
            for i in range(2):
                seq_img[i] = image_utils.normalize(sequence_img[i].convert("RGB"))
            return seq_img, ratio, bbox_sequence_square
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for i in range(2):
                sequence_img[i].save(os.path.join(save_path, "frame_{:02d}.png".format(i)))

    def __getitem__(self, index):
        seq_img_path = self.sequence_data["img_full_path"][index]
        cad_name = self.sequence_data["cad_name"][index]
        seq_img, ratio, bbox_sequence_square = self._fetch_sequence(seq_img_path)
        data_batch = {names[i]: [] for i in range(1, len(names))}
        for name in names[2:]:
            tmp = self.sequence_data[name][index]
            if name == "delta_uv":
                tmp = tmp * ratio / (self.image_size / 2)
            elif name == "delta_depth":
                tmp = tmp * ratio
            data_batch[name] = torch.from_numpy(np.ascontiguousarray(tmp)).float()
        seq_img = torch.from_numpy(np.ascontiguousarray(seq_img)).float()
        ratio = torch.from_numpy(np.ascontiguousarray(ratio)).float()
        bbox_sequence_square = torch.from_numpy(np.ascontiguousarray(bbox_sequence_square)).float()
        data_batch = dict(seq_img=seq_img,
                          ratio=ratio,
                          cad_name=cad_name,
                          bbox_sequence_square=bbox_sequence_square,
                          delta_rotation=data_batch["delta_rotation"],
                          delta_uv=data_batch["delta_uv"],
                          delta_depth=data_batch["delta_depth"],
                          rotation_first_frame=data_batch["rotation_first_frame"],
                          uv_first_frame=data_batch["uv_first_frame"],
                          depth_first_frame=data_batch["depth_first_frame"],
                          gt_rotations=data_batch["gt_rotations"],
                          gt_translations=data_batch["gt_translations"],
                          img_path=seq_img_path[0])
        return data_batch

    def save_random_sequences(self):
        print_and_log_info(self.logger, "Saving training samples at {}".format(self.save_path))
        list_index = np.unique(np.random.randint(0, self.__len__(), 10))
        for index in list_index:
            save_sequence_path = os.path.join(self.save_path, "{:06d}".format(index))
            self._fetch_sequence(self.sequence_data["img_full_path"][index], save_sequence_path)