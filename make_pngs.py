"""make_pngs.py

Makes a PNG equivalent for every nanoparticle image

William Thornley Oct 2022
"""

import pathlib
import glob

import tqdm
import numpy as np
import hyperspy .api as hs
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt

project_dir = pathlib.Path().resolve()
image_glob = project_dir / '*' / '*.emi'  # <-- EDIT THIS!!!
images_filename_strings = glob.glob(str(image_glob), recursive=True)

# for image_filename in images_filename_strings:
#     print(image_filename)    
# print(image_glob)
# exit(0)

def plot(image_data: np.array, pixel_scale: float, units: str):

    fig = plt.figure(dpi=300, frameon=False, figsize=(8, 8))
    ax = fig.add_axes(rect=(0, 0, 1, 1))
    ax.axis('off')
    ax.imshow(image_data, interpolation='none', cmap='inferno')
    scalebar = ScaleBar(pixel_scale, units, length_fraction=0.25, color='w', box_alpha=0)
    ax.add_artist(scalebar)
    ax.autoscale_view(tight=True)
    # plt.show(block=True)
    return fig


for image_filename in tqdm.tqdm(images_filename_strings):
    if 'search' in image_filename.lower():  # Only care about acquires, not searches.
        continue
    if 'SI HAADF' in image_filename:
        continue  # This is an EDX map
    savename = image_filename[:-4] + '.png'
    image = hs.load(image_filename)
    try:
        px_scale = image.axes_manager['x'].scale
    except AttributeError:
        print(f"Cannot open {image_filename}")
        continue
    units = image.axes_manager['x'].units
    figure = plot(image.data, px_scale, units)
    figure.savefig(savename, dpi=300)
    plt.close(figure)

