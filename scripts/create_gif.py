import os.path
import shutil

import imageio
import matplotlib.pyplot as plt
import numpy as np


def create_gif(
    time_series: np.ndarray,
    saving_directory: str,
    name_file: str,
    delete_imgs: bool = False,
    normalize: bool = True,
):
    time_series_min = time_series.min()
    time_series_max = time_series.max()
    if time_series.ndim > 3:
        raise ValueError("Error: The time series should be (time, height, width)")
    if not os.path.exists(saving_directory + "/img_for_gif"):
        os.makedirs(saving_directory + "/img_for_gif")
    images = []
    cmap = "magma"  #'RdBu_r' #'viridis'
    for i in range(time_series.shape[0]):
        if normalize:
            plt.imshow(
                time_series[i],
                origin="lower",
                cmap=cmap,
                vmin=time_series_min,
                vmax=time_series_max,
            )
        else:
            plt.imshow(time_series[i], cmap=cmap, origin="lower")
        plt.axis("off")
        plt.savefig(
            saving_directory + f"/img_for_gif/time_series_{i}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
        images.append(
            imageio.imread(saving_directory + f"/img_for_gif/time_series_{i}.png")
        )

    imageio.mimsave(saving_directory + "/" + name_file + ".gif", images, duration=0.1)
    if delete_imgs:
        shutil.rmtree(saving_directory + "/img_for_gif")
