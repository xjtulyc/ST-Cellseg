import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq
from cellpose import models


def cellpose(img, min_size=15):
    model = models.Cellpose(model_type='nuclei')
    res, _, _, _ = model.eval(
        img,
        channels=[0, 0],
        diameter=None,
        min_size=min_size,
    )
    return res


img = sq.datasets.visium_fluo_image_crop()
crop = img.crop_corner(1000, 1000, size=1000)
# crop.show(channelwise=True)
crop.show()
plt.show()
sq.im.segment(img=crop, layer="image", channel=0, method=cellpose)

print(crop)
print(f"Number of segments in crop: {len(np.unique(crop['segmented_custom']))}")

# fig, axes = plt.subplots(1, 2, figsize=(10, 20))
fig, axes = plt.subplots(1, 2)
crop.show("image", channel=0, ax=axes[0])
_ = axes[0].set_title("DAPI")
crop.show("segmented_custom", cmap="jet", interpolation="none", ax=axes[1])
_ = axes[1].set_title("Cellpose segmentation")
plt.show()
sq.im.segment(img=crop, layer="image", channel=0, method=cellpose, min_size=200)

print(crop)
print(f"Number of segments in crop: {len(np.unique(crop['segmented_custom']))}")

fig, axes = plt.subplots(1, 2, figsize=(10, 20))
crop.show("image", channel=0, ax=axes[0])
_ = axes[0].set_title("DAPI")
crop.show("segmented_custom", cmap="jet", interpolation="none", ax=axes[1])
_ = axes[1].set_title("Cellpose segmentation")


def cellpose_he(img, min_size=15, flow_threshold=0.4, channel_cellpose=0):
    model = models.Cellpose(model_type='nuclei')
    res, _, _, _ = model.eval(
        img,
        channels=[channel_cellpose, 0],
        diameter=None,
        min_size=min_size,
        invert=True,
        flow_threshold=flow_threshold,
    )
    return res


img = sq.datasets.visium_hne_image_crop()
crop = img.crop_corner(0, 0, size=1000)
crop.show("image", channelwise=True)

sq.im.segment(img=crop, layer="image", channel=None, method=cellpose_he, channel_cellpose=1)

print(crop)
print(f"Number of segments in crop: {len(np.unique(crop['segmented_custom']))}")

fig, axes = plt.subplots(1, 2, figsize=(10, 20))
crop.show("image", channel=None, ax=axes[0])
_ = axes[0].set_title("H&E")
crop.show("segmented_custom", cmap="jet", interpolation="none", ax=axes[1])
_ = axes[1].set_title("Cellpose segmentation")

sq.im.segment(img=crop, layer="image", channel=None, method=cellpose_he, flow_threshold=0.8, channel_cellpose=1)

print(crop)
print(f"Number of segments in crop: {len(np.unique(crop['segmented_custom']))}")

fig, axes = plt.subplots(1, 2, figsize=(10, 20))
crop.show("image", channel=None, ax=axes[0])
_ = axes[0].set_title("H&E")
crop.show("segmented_custom", cmap="jet", interpolation="none", ax=axes[1])
_ = axes[1].set_title("Cellpose segmentation")

sq.im.segment(img=crop, layer="image", channel=None, method=cellpose_he, channel_cellpose=0)

print(crop)
print(f"Number of segments in crop: {len(np.unique(crop['segmented_custom']))}")

fig, axes = plt.subplots(1, 2, figsize=(10, 20))
crop.show("image", channel=None, ax=axes[0])
_ = axes[0].set_title("H&E")
crop.show("segmented_custom", cmap="jet", interpolation="none", ax=axes[1])
_ = axes[1].set_title("Cellpose segmentation")

sq.im.segment(img=crop, layer="image", channel=None, method=cellpose_he, flow_threshold=0.8, channel_cellpose=0)

print(crop)
print(f"Number of segments in crop: {len(np.unique(crop['segmented_custom']))}")

fig, axes = plt.subplots(1, 2, figsize=(10, 20))
crop.show("image", channel=None, ax=axes[0])
_ = axes[0].set_title("H&E")
crop.show("segmented_custom", cmap="jet", interpolation="none", ax=axes[1])
_ = axes[1].set_title("Cellpose segmentation")
