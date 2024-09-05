import glob
import os
import re
import shutil


def create_gif(time_series, saving_directory, name_file="density", delete_imgs=True):
    import imageio
    import matplotlib.pyplot as plt

    time_series_min = time_series.min()
    time_series_max = time_series.max()
    if time_series.ndim > 3:
        print("Error: The time series should be (time, height, width)")
        return
    if not os.path.exists(saving_directory + "/img_for_gif"):
        os.makedirs(saving_directory + "/img_for_gif")
    for i in range(time_series.shape[0]):
        plt.imshow(
            time_series[i], origin="lower", vmin=time_series_min, vmax=time_series_max
        )
        plt.axis("off")
        plt.savefig(
            saving_directory + f"/img_for_gif/time_series_{i}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
    images = []
    for file_name in sorted(
        glob.glob(saving_directory + "/img_for_gif/*.png"),
        key=lambda x: int(re.findall(r"\d+", x)[1]),
    ):
        images.append(imageio.imread(file_name))
    imageio.mimsave(saving_directory + "/" + name_file + ".gif", images, duration=0.1)
    if delete_imgs:
        shutil.rmtree(saving_directory + "/img_for_gif")
