# -*- coding: utf-8 -*-
"""
https://github.com/ylabbe/cosypose/blob/master/cosypose/datasets/augmentations.py
https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py
"""
from PIL import ImageEnhance, ImageFilter, Image
import numpy as np
import random
import cv2


class Erosion:
    def __init__(self, kernel_size=3, iterations=2, p=0.5):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.p = p
        self.iterations = iterations

    def __call__(self, list_imgs):
        if random.random() <= self.p:
            for i in range(len(list_imgs)):
                img = np.copy(list_imgs[i])
                img = cv2.erode(img, self.kernel, self.iterations)
                list_imgs[i] = img
        return list_imgs


class PillowRGBAugmentation:
    def __init__(self, len_sequences, pillow_fn, p, factor_interval, delta_factor_interval):
        self._pillow_fn = pillow_fn
        self.p = p
        self.factor_interval = factor_interval
        self.delta_factor_interval = (delta_factor_interval[0], delta_factor_interval[1], len_sequences)

    def __call__(self, list_imgs):
        if random.random() <= self.p:
            init_factor = random.uniform(*self.factor_interval)
            delta_factor = np.random.uniform(*self.delta_factor_interval)
            for i in range(len(list_imgs)):
                list_imgs[i] = self._pillow_fn(list_imgs[i]).enhance(factor=init_factor * (1 + delta_factor[i]))
        return list_imgs


class PillowSharpness(PillowRGBAugmentation):
    def __init__(self, len_sequences, p=0.3, factor_interval=(0, 40.),
                 delta_factor_interval=(0.0, 0.4)):
        super().__init__(pillow_fn=ImageEnhance.Sharpness,
                         p=p,
                         factor_interval=factor_interval,
                         len_sequences=len_sequences,
                         delta_factor_interval=delta_factor_interval)


class PillowContrast(PillowRGBAugmentation):
    def __init__(self, len_sequences, p=0.3, factor_interval=(0.5, 1.6),
                 delta_factor_interval=(0.0, 0.4)):
        super().__init__(pillow_fn=ImageEnhance.Contrast,
                         p=p,
                         factor_interval=factor_interval,
                         len_sequences=len_sequences,
                         delta_factor_interval=delta_factor_interval)


class PillowBrightness(PillowRGBAugmentation):
    def __init__(self, len_sequences, p=0.5, factor_interval=(0.5, 2.0),
                 delta_factor_interval=(0.0, 0.4)):
        super().__init__(pillow_fn=ImageEnhance.Brightness,
                         p=p,
                         factor_interval=factor_interval,
                         len_sequences=len_sequences,
                         delta_factor_interval=delta_factor_interval)


class PillowColor(PillowRGBAugmentation):
    def __init__(self, len_sequences, p=1, factor_interval=(0.0, 20.0),
                 delta_factor_interval=(0.0, 0.4)):
        super().__init__(pillow_fn=ImageEnhance.Color,
                         p=p,
                         factor_interval=factor_interval,
                         len_sequences=len_sequences,
                         delta_factor_interval=delta_factor_interval)


class PillowRotation:
    def __init__(self, p=0.8, rotation_interval=(0, 360)):
        if random.uniform(0, 1) > p:  # apply inplane rotation
            self.degrees = random.randint(*rotation_interval)
        else:
            self.degrees = 0

    def __call__(self, list_imgs):
        if self.degrees != 0:
            for i in range(len(list_imgs)):
                list_imgs[i] = list_imgs[i].rotate(self.degrees)
        return list_imgs, self.degrees


class PillowBlur:
    def __init__(self, p=0.4, factor_interval=(1, 3)):
        self.p = p
        self.k = random.randint(*factor_interval)

    def __call__(self, list_imgs):
        for i in range(len(list_imgs)):
            list_imgs[i] = list_imgs[i].filter(ImageFilter.GaussianBlur(self.k))
        return list_imgs


def apply_data_augmentation(len_sequences, sequence_img):  # Same as cosypose
    # to avoid artifact on background, to use the mask of input image
    sequence_mask = [sequence_img[i].getchannel("A") for i in range(len_sequences)]
    sharpness = PillowSharpness(len_sequences, factor_interval=(0, 5.))
    contrast = PillowContrast(len_sequences, factor_interval=(0.2, 5.))
    brightness = PillowBrightness(len_sequences, factor_interval=(0.1, 1.5))
    color = PillowColor(len_sequences, factor_interval=(0.0, 3))
    blur = PillowBlur()
    img_aug = sharpness(sequence_img)
    img_aug = contrast(img_aug)
    img_aug = brightness(img_aug)
    img_aug = color(img_aug)
    img_aug = blur(img_aug)

    black_img = Image.new("RGB", img_aug[0].size)
    for i in range(len_sequences):
        img_aug[i] = Image.composite(img_aug[i], black_img, sequence_mask[i])
    return img_aug


class NumpyGaussianNoise:
    def __init__(self, p, factor_interval=(0.01, 0.3)):
        self.noise_ratio = random.uniform(*factor_interval)
        self.p = p

    def __call__(self, list_imgs):
        for i in range(len(list_imgs)):
            if random.random() <= self.p:
                img = np.copy(list_imgs[i])
                noisesigma = random.uniform(0, self.noise_ratio)
                gauss = np.random.normal(0, noisesigma, img.shape) * 255
                img = img + gauss

                img[img > 255] = 255
                img[img < 0] = 0
                list_imgs[i] = Image.fromarray(np.uint8(img))
        return list_imgs


def apply_data_augmentation_wo_mask(len_sequences, sequence_img):  # Same as cosypose
    # to avoid artifact on background, to use the mask of input image
    masks = [np.asarray(img).astype(np.float32).sum(-1) for img in sequence_img]
    for mask in masks:
        mask[mask > 0] = 1
    sharpness = PillowSharpness(len_sequences, factor_interval=(0, 5.))
    contrast = PillowContrast(len_sequences, factor_interval=(0.2, 5.))
    brightness = PillowBrightness(len_sequences, factor_interval=(0.1, 1.5))
    color = PillowColor(len_sequences, factor_interval=(0.0, 3))
    gaussian_noise = NumpyGaussianNoise(p=0.2, factor_interval=(0.1, 0.04))
    blur = PillowBlur()

    img_aug = sharpness(sequence_img)
    img_aug = contrast(img_aug)
    img_aug = brightness(img_aug)
    img_aug = color(img_aug)
    img_aug = blur(img_aug)
    img_aug = gaussian_noise(img_aug)

    black_img = Image.new("RGB", img_aug[0].size)
    for i in range(len_sequences):
        mask = Image.fromarray((255 * masks[i]).astype(np.uint8)).convert('L')
        img_aug[i] = Image.composite(img_aug[i], black_img, mask)
    return img_aug


def apply_data_augmentation_wo_mask_w_erosion(len_sequences, sequence_img):  # Same as cosypose
    # to avoid artifact on background, to use the mask of input image
    masks = [np.asarray(img).astype(np.float32).sum(-1) for img in sequence_img]
    for mask in masks:
        mask[mask > 0] = 1
    sharpness = PillowSharpness(len_sequences, factor_interval=(0, 5.))
    contrast = PillowContrast(len_sequences, factor_interval=(0.2, 5.))
    brightness = PillowBrightness(len_sequences, factor_interval=(0.1, 1.5))
    color = PillowColor(len_sequences, factor_interval=(0.0, 3))
    gaussian_noise = NumpyGaussianNoise(p=0.2, factor_interval=(0.1, 0.04))
    blur = PillowBlur()
    erosion = Erosion()

    img_aug = sharpness(sequence_img)
    img_aug = contrast(img_aug)
    img_aug = brightness(img_aug)
    img_aug = color(img_aug)
    img_aug = blur(img_aug)
    img_aug = gaussian_noise(img_aug)

    masks = erosion(masks)

    black_img = Image.new("RGB", img_aug[0].size)
    for i in range(len_sequences):
        mask = Image.fromarray((255 * masks[i]).astype(np.uint8)).convert('L')
        img_aug[i] = Image.composite(img_aug[i], black_img, mask)
    return img_aug