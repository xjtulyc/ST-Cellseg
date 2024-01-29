# Import the StarDist 2D segmentation models.
import matplotlib.pyplot as plt
import numpy as np
# Import squidpy and additional packages needed for this tutorial.
import squidpy as sq
# Import the recommended normalization technique for stardist.
from csbdeep.utils import normalize
from stardist.models import StarDist2D

StarDist2D.from_pretrained()

# Load the image and visualize its channels.
img = sq.datasets.visium_fluo_image_crop()
crop = img.crop_corner(1000, 1000, size=1000)
crop.show()

StarDist2D.from_pretrained('2D_versatile_fluo')


def stardist_2D_versatile_fluo(img, nms_thresh=None, prob_thresh=None):
    # Make sure to normalize the input image beforehand or supply a normalizer to the prediction function.
    # this is the default normalizer noted in StarDist examples.
    img = normalize(img, 1, 99.8, axis=(0, 1))
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    labels, _ = model.predict_instances(img, nms_thresh=nms_thresh, prob_thresh=prob_thresh)
    return labels


sq.im.segment(
    img=crop,
    layer="image",
    channel=0,
    method=stardist_2D_versatile_fluo,
    layer_added='segmented_stardist',
    nms_thresh=None,
    prob_thresh=None
)

# Plot the DAPI channel of the image crop and the segmentation result.
print(crop)
print(f"Number of segments in crop: {len(np.unique(crop['segmented_stardist']))}")

fig, axes = plt.subplots(1, 2)
crop.show("image", channel=0, ax=axes[0])
_ = axes[0].set_title("DAPI")
crop.show("segmented_stardist", cmap="jet", interpolation="none", ax=axes[1])
_ = axes[1].set_title("segmentation")

plt.show()
