
import os

from .io import get_target_hostcutoutpath

def get_target_cutoutimg(name, ax=None, release="dr3", clean_axis=True,
                         **kwargs):
    """Fetch the host cutout image for a given target."""
    img_path = get_target_hostcutoutpath(name, release=release)
    # nothing to show. nothing done.
    if img_path is None or not os.path.isfile(img_path):
        return None

    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    img_ = plt.imread(img_path)
    _ = ax.imshow(img_, **kwargs)

    if clean_axis:
        ax.set_axis_off()

    return fig
