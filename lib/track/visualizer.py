import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2022)
COLORS_SPACE = np.random.randint(0, 255, size=(1000, 3))


def vis_bbox_and_mask(img, xyxy_boxes=None, masks=None, colors=None, thick=3, blend=0.1):
    """Visualizes a bounding box."""
    img = img.astype(np.uint8)
    if xyxy_boxes is not None:
        for idx, box in enumerate(xyxy_boxes):
            if colors is None:
                color = (255, 0, 0) # COLORS_SPACE[idx]
            elif colors == "COLORSPACE":
                color = list(COLORS_SPACE[idx])
                color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness=thick)
    if masks is not None:
        for idx, mask in enumerate(masks):
            if colors is None:
                color = (255, 0, 0) # COLORS_SPACE[idx]
            elif colors == "COLORSPACE":
                color = list(COLORS_SPACE[idx])
                color = (int(color[0]), int(color[1]), int(color[2]))
            mask = mask.astype(np.bool)
            img[mask, 0] = img[mask, 0] * blend + (1 - blend) * color[0]
            img[mask, 1] = img[mask, 1] * blend + (1 - blend) * color[1]
            img[mask, 2] = img[mask, 2] * blend + (1 - blend) * color[2]
    return img


def mask_flow(flow, masks):
    mask_obj = np.zeros_like(masks[0], dtype=np.bool)
    for idx, mask in enumerate(masks):
        mask = mask.astype(np.bool)
        mask_obj[mask] = True
    flow[np.invert(mask_obj), :] = 0
    return flow


def plot_tracking(imgs, titles, save_path):
    plt.figure(figsize=(15, 5))
    for idx_plot in range(3):
        plt.subplot(1, 3, idx_plot+1)
        if imgs[idx_plot] is not None:  
            plt.imshow(imgs[idx_plot])
            plt.title(titles[idx_plot])
            plt.axis("off")
        else:
            black_img = np.ones_like(imgs[0])*255
            black_img = black_img.astype(np.uint8)
            plt.imshow(black_img)
            plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', dpi=100, pad_inches=0.05)
    plt.close("all")