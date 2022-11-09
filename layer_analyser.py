"""layer_analyser.py

Allows analysing of layer thickness in the samples.

Provides a matplotlib GUI to allow for the adjusting of threshold values used to segment images, by pixel intensities, 
into regions of different thicknesses. Steps the user through every image for every sample in the dataset. Creates a 
results directory where every source-datafile is symbolically linked to and all the results of this program are saved. 
The results are saved, for each image, as a PNG containing a display of the thresholded image and the threshold values, 
as well as a JSON file containing the threshold values selected and the numbers of pixels in each region. For each sample, 
results are saved (also in JSON) containing the total areas containing certain thicknesses. 

Notes: 
It should be noted that the program will ignore images outside a certain magnification range. 
The figures have been arranged so that they can take one side of your screen while the terminal can take the other side. 
If the segmentation map looks weird, it could be that you have two lines hidden ontop of eachother. To fox you should delete 
threshold lines until the problem is resolved.

To the best of my knowledge Sample#3=Sample#9, Sample#6=Sample#10 and Sample#8=Sample#12. For ease of programming I will
call them samples A, B and C respectively.
"""

from __future__ import annotations
import glob
import json
from pathlib import Path
from enum import Enum
from typing import Generator, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.axes
import matplotlib.text

import numpy as np
import scipy.ndimage
import scipy.signal

import hyperspy.api as hs
from hyperspy._signals.signal2d import Signal2D as hsSignal2DType

############################
# Parameters & DEFINITIONS #
############################

project_root = Path().resolve()  # Should be .../PtSe_G_Duesberg/

MAX_SCALE = 0.07  # nm per pixel
MIN_SCALE = 0.02  # nm per pixel

SAMPLE_A_GLOBS = (
    project_root / 'EPSIC Data' / 'EPSICOct21' / 'Day2' / 'PtSe Sample 3' / 'HAADF_*.dm3',
    project_root / 'EPSIC Data' / 'EPSICOct21' / 'Day3' / 'PtSe Sample 3 Revisit' / 'HAADF_*.dm3',
    project_root / 'EPSIC Data' / 'EPSICMar22' / 'Day2' / 'PtSe2 N0.9' / 'HAADF_*.dm3'
)

SAMPLE_B_GLOBS = (
    project_root / 'EPSIC Data' / 'EPSICOct21' / 'Day2' / 'PtSe Sample 6' / 'HAADF_*.dm3',
    project_root / 'EPSIC Data' / 'EPSICMar22' / 'Day2' / 'Pt Se2 No.10' / 'HAADF_*.dm3'
)

SAMPLE_C_GLOBS = (
    project_root / 'EPSIC Data' / 'EPSICOct21' / 'Day2' / 'PtSe Sample 8' / 'HAADF_*.dm3',
    project_root / 'EPSIC Data' / 'EPSICMar22' / 'Day4' / 'PtSe2 No.12' / 'HAADF_*.dm3'
)

SAVE_LOCATION = project_root / 'Analysis_Scripts' / 'Quick Layer Analysis' / 'results'  # Continue from {EPSICOct21 or EPSICMar22} / Day...


def setup_save_location(filename: Path) -> Path:
    """Takes the filename of the image being processed. Generates filenames for where the corresponding results, for 
    that image, should be saved.
    """

    save_name = SAVE_LOCATION.joinpath(*filename.parts[-4:])  # Should be SAVE_LOCATION / {EPSICOct21 or EPSICMar22} / ...

    if not save_name.parent.exists():
        save_name.parent.mkdir(parents=True)  # Creates the tower of parent directories if they don't exist

    return save_name 


class SampleType(Enum):
    """Encodes the type of sample.
    
    Specifies the type of sample according to:
    A: Sample#3/#9
    B: Sample#6/#10
    C: Sample#8/#12
    """
    A, B, C = range(1, 4)


class ImageAction(Enum):
    """Encodes the result states for the `refine_lines` function."""
    REFINED, DISCARD, QUIT = range(1, 4)


class LayerThicknessResult(dict):
    """Determined areas of certain layer-thicknesses for a sample.
    
    A type of dict where the keys are integers of the number of atomic layers and the values are the area in nanomaters, 
    found to have that thickness. Some may have `intensity_thresholds` as a key too, when they represent the thresholding 
    results from a single image, however results from an entire sample won't have this.
    """

    def __init__(self, units='px'):
        self.units = units
        super().__init__()

    @classmethod
    def from_lists(cls, layer_thicknesses: Iterable, pixel_counts: Iterable, thresholds: List) -> LayerThicknessResult:
        """Generates a result, in units of pixels, from lists such as those returned by np.unique()
        
        Generates a LayerThicknessResult from iterables of layer-thicknesses, the number of pixels within those segmented layer-thickness 
        areas, and the intensity thresholds used to segment the layer-thickness regions.
        """

        result = cls()
        for layer_thickness, pixel_count in zip(layer_thicknesses, pixel_counts):
            layer_thickness = int(layer_thickness)
            pixel_count = int(pixel_count)
            result[layer_thickness] = pixel_count
        result['intensity_thresholds'] = thresholds
        return result

    def convert_units(self, units_per_pixel: float, units_name: str) -> None:
        """Converts units"""

        for key, area in self.items():
            if not isinstance(key, int): continue  # Must be a different attribute
            area *= units_per_pixel
        self.units = units_name

    def save(self, filename: Path):
        """Saves the results as a JSON file"""

        self['units'] = self.units  # Must be in the dict to be saved!
        try:
            with open(filename, 'x') as file:
                json.dump(self, file, indent='\t')
        except FileExistsError:  # Replace the error with something more meaningful
            raise FileExistsError(f"Cannot save LayerThicknessResult because the filename {filename} already exists.")

    @classmethod
    def load(cls, filename: Path) -> LayerThicknessResult:
        """Loads a LayerThicknessResult from JSON file."""

        with open(filename) as file:
            json_dict = json.load(file)
        
        new_units = json_dict['units']
        new_result = cls(new_units)
        for key in json_dict:
            if key.isnumeric():
                new_result[int(key)] = json_dict[key]
            else:
                new_result[key] = json_dict[key]

        return new_result


    def __iadd__(self, other) -> LayerThicknessResult:
        """Adds the two results"""

        if other.units != self.units:
            raise ValueError("The results cannot be added unless they are of the same units!")
        for thickness, area in other.items():
            if self.get(thickness) is None:
                self[thickness] = area
            else:
                self[thickness] += area

        return self    

    def __repr__(self) -> str:
        repr_string = "LayerThicknessResult:\n"
        for key in self:
            if isinstance(key, int):
                repr_string += f"\t{key} layer area: {self[key]:.2E} {self.units}^2\n"
            else:
                repr_string += f"\t{key} : {self[key]}\n"
        return repr_string + '\n'


class SampleData:
    """Finds all the filenames for a sample-type and defines a generator for accessing the corresponding hyperspy signals."""

    def __init__(self, sample_type: SampleType, filename_globs: Iterable[Path]) -> None:
        self.sample_type: SampleType = sample_type
        self.filenames: Iterable[Path] = self.find_filenames(filename_globs)

    def __repr__(self) -> str:
        return f"SampleData({self.sample_type}, [{self.filenames[0]}, ...] ({len(self.filenames)} filenames))"

    @staticmethod
    def find_filenames(glob_paths: Iterable[Path]) -> List[Path]:
            """Processes glob or globs in the format of a pathlib Path. Finds all filenames."""

            if len(glob_paths) == 0:
                raise ValueError("glob_paths parameter should contain something.")

            globs = [str(path) for path in glob_paths]
            filenames = []
            for filename_glob in globs:
                globbed_filenames = glob.glob(filename_glob)
                globbed_paths = [Path(filename) for filename in globbed_filenames]
                filenames.extend(globbed_paths)

            return filenames

    def images_generator(self) -> Generator[Tuple[hsSignal2DType, Path], None, None]:
        """Loads images from the filenames one by one, skipping images at an inappropriate magnification."""

        for path in self.filenames:
            signal = hs.load(path)
            # Filter images based on magnification
            if not signal.axes_manager['x'].scale == signal.axes_manager['y'].scale or \
               not MIN_SCALE < signal.axes_manager['x'].scale < MAX_SCALE:
               continue
            yield signal, path


class DraggableVLine:
    """A vertical matplotlib line that can be dragged with the mouse independently of others of the same type. Attaches a text label."""

    def __init__(self, hist_ax: matplotlib.axes.Axes, resegment_callback, coord: float, color: str):

        self.ax = hist_ax
        self.resegment_callback = resegment_callback
        self.canvas = hist_ax.get_figure().canvas
        self.coord = coord  # The x-axis coordinate value

        x = [coord, coord]
        y = self.ax.get_ylim()  # Can set to preference
 
        self.line = matplotlib.lines.Line2D(x, y, color=color, picker=2.5)  # The clickable width is picker
        self.ax.add_line(self.line)
        self.canvas.draw_idle()
        self.sid = self.canvas.mpl_connect('pick_event', self.clickonline)

    def clickonline(self, event):
        if event.artist == self.line:
            self.follower = self.canvas.mpl_connect("motion_notify_event", self.followmouse)
            self.releaser = self.canvas.mpl_connect("button_press_event", self.releaseonclick)

    def followmouse(self, event):
        if event.xdata == None:
            print("WARN: event.xdata is None")
        self.line.set_xdata([event.xdata, event.xdata])
        self.canvas.draw_idle()

    def releaseonclick(self, event):
        self.coord = self.line.get_xdata()[0]
        self.canvas.mpl_disconnect(self.releaser)
        self.canvas.mpl_disconnect(self.follower)
        self.resegment_callback()

    def remove_from_fig(self):
        self.line.remove()
        self.canvas.draw_idle()


def image_smoother(image_data: np.ndarray, pixel_scale: float, sigma: float = 0.2) -> np.ndarray:
    """Applied gaussian smoothing to the image with a sigma fixed in nanometers.
    
    Parameters
    ----------
    image_data : np.ndarray
        The image to be smoothed as a 2D array of ints
    pixel_scale : float
        The size of the pixels in the image, in nanomaters
    sigma : float
        The standard deviation of the gaussian blur, in nanometers
        
    Returns
    -------
    np.ndarray
        The blurred image. Note: Converts into floating-point datatype!
    """

    sigma_px = sigma / pixel_scale
    smoothed_image = scipy.ndimage.gaussian_filter(image_data.astype(float), sigma_px)
    return smoothed_image


def initial_threshold_estimate(image: np.ndarray) -> List[float]:
    """Makes initial estimates of threshold positions.
    
    Generates some (bad) initial estimates of the number and location of thresholds by finding local minima in the histogram of pixel intensities.
    """

    histogram, histogram_bin_edges = np.histogram(image.flatten(), bins=150)

    minima = scipy.signal.argrelmin(histogram, order=5)[0]
    bounds = []  # Contains the boundries as indices of the histogram_bin_edges

    if len(minima) < 2:
        bounds.append((histogram_bin_edges[0], histogram_bin_edges[int(len(histogram_bin_edges)//2)]))
        bounds.append(histogram_bin_edges[int(len(histogram_bin_edges)//2)], histogram_bin_edges[-1])
        return [bound[0] for bound in bounds]

    for start, end in zip(minima, minima[1:]):  # the nth and (n+1)th local minuma found
        bound_start = histogram_bin_edges[start]  # The x-value at the nth index in the histogram
        bound_end = histogram_bin_edges[end]  # '' for n+1
        bounds.append((bound_start, bound_end))
    bounds.append((bound_end, histogram_bin_edges[-1]))  # Capture the tail of the histogram

    return [bound[0] for bound in bounds]


def segment_image(image: np.ndarray, thresholds: List) -> np.ndarray:
    """Takes image and thresholds as input. Segments the image into regions matching those thresholds. Segmented image 
    contains integer values corresponding to the segmenation.
    """

    segmented_image = np.zeros(image.shape, dtype=int)
    if len(thresholds) == 0: return segmented_image  # No thresholds mean it must all be background

    thresholds = [0,] + thresholds + [np.max(image),]  # Prepend 0 so that the 'background' is the 0th threshold region
    for threshold_index, (threshold_low, threshold_high) in enumerate(zip(thresholds, thresholds[1:])):
        threshold_region = np.logical_and(threshold_low < image, image < threshold_high)
        segmented_image[threshold_region] = threshold_index
    
    return segmented_image


def refine_lines(image: np.ndarray, current_thresholds: Iterable, save_name: Union[Path, None]) -> Tuple[ImageAction, Union[None, List[float]]]:
    """A GUI to allow refinement of the threshold lines.
    
    Takes the histogram of intensities and current estimates of the threshold values and displays them to the user. The
    user can adjust these threshold values graphically, including adding and removing thresholds, until satisfied.
    
    Parameters
    ----------
    histogram : np.ndarray
        An Nx2 dimension array containing the ((x, y), ...) values (x being intensity and y being frequency) representing 
        the histogram of intensities.
    current_thresholds : Iterable
        A 1D iterable containing all the current threshold values as intensities
    savename : Path
        A Python path specifying the name to save the segmentation figure and JSON LayerThicknessResult. Should not have a suffix 
        (anything after the fullstop)

    Returns
    -------
    Tuple[ImageAction, Union[None, List[float]]]
        An ImageAction enum encoding the state the function exited in and either a list of the thresholds or None
    """

    PLOT_DPI = 210
    plot_title = save_name.with_suffix('').parts[-4:]
    plot_title = ' '.join(plot_title)

    # First plot the display graph
    fig = plt.figure(figsize=(4, 4), dpi=PLOT_DPI)  # For some reason the figsize doesn't do what it says...
    image_axis = plt.subplot2grid((2, 2), (0, 0), 1, 1, fig)
    thresholded_axis = plt.subplot2grid((2, 2), (0, 1), 1, 1, fig)
    hist_axis = plt.subplot2grid((2, 2), (1, 0), 1, 2, fig)

    hist_axis.ticklabel_format(style='sci', scilimits=(-1, 2))
    hist_axis.set_xlabel("Intensity (pixel counts)")
    hist_axis.set_ylabel("Frequency")
    image_axis.axes.get_xaxis().set_visible(False)
    image_axis.axes.get_yaxis().set_visible(False)
    thresholded_axis.axes.get_xaxis().set_visible(False)
    thresholded_axis.axes.get_yaxis().set_visible(False)

    image_axis.imshow(image)
    hist_axis.hist(image.flatten(), bins=150, cumulative=False, histtype='step')
    threshold_image = thresholded_axis.imshow(segment_image(image, list()))

    fig.suptitle(plot_title)
    fig.show()

    vlines: List[DraggableVLine] = []
    def resegment_callback():
        """Updates the threshold image in the figure. Also sorts the lines"""
        fresh_segmentation = segment_image(image, [vline.coord for vline in vlines])
        threshold_image.set_data(fresh_segmentation)
        threshold_image.set(clim=(0, len(vlines)))
        fig.canvas.draw_idle()
        vlines.sort(key=lambda vline: vline.coord)

    for vline_coord in current_thresholds:
        vline = DraggableVLine(hist_axis, resegment_callback, vline_coord, 'r')
        vlines.append(vline)
    resegment_callback()

    new_vline_spawnpoint = np.mean(image)

    prompt = "Add another threshold (a), remove a threshold (d), re-initialise thresholds (r), save thresholds and continue to next (s), \n" + \
    "discard this image (x) or quit (q)."
    print(prompt)
    while True:
        response = input("Enter command: ")
        if response == 'a':
            new_vline = DraggableVLine(hist_axis, resegment_callback, new_vline_spawnpoint, 'r')
            vlines.append(new_vline)
            resegment_callback()
        elif response == 'd':
            if len(vlines) == 0:
                print("There are no more thresholds to remove.")
            else:
                vlines[-1].remove_from_fig()
                vlines.pop()  # Remove the last vline, user can re-arrange themselves
                resegment_callback()
        elif response == 'r':
            for vline in vlines: vline.remove_from_fig()
            vlines = [DraggableVLine(hist_axis, resegment_callback, vline_coord, 'r') for vline_coord in initial_threshold_estimate(image)]
            resegment_callback()
        elif response == 's':
            if save_name is not None:
                fig.savefig(save_name, dpi=PLOT_DPI)
            plt.close(fig)
            return ImageAction.REFINED, [vline.coord for vline in vlines]
        elif response == 'x': 
            plt.close(fig)
            return ImageAction.DISCARD, None
        elif response == 'q': 
            plt.close(fig)
            return ImageAction.QUIT, None
        else:
            print("Response not recognised.")
            print(prompt)


# Defines samples, finds all the corresponding filenames
sampleA = SampleData(SampleType.A, SAMPLE_A_GLOBS)
sampleB = SampleData(SampleType.B, SAMPLE_B_GLOBS)
sampleC = SampleData(SampleType.C, SAMPLE_C_GLOBS)

sample_result_save_names = {
    sampleA : SAVE_LOCATION / 'sampleA.json',
    sampleB : SAVE_LOCATION / 'sampleB.json',
    sampleC : SAVE_LOCATION / 'sampleC.json'
}


##############
# Processing #
##############

if __name__ == '__main__':

    print(
        "\nBeginning analysis, there are approximately", 
        int(sum([len(sample.filenames) for sample in sample_result_save_names.keys()])*0.9),  # Guess 10% invalid files
        "valid files in", len(sample_result_save_names.keys()), "sample catagories.\n"
    )
    image_tally = 1

    for sample in (sampleA, sampleB, sampleC):

        print("New sample ----------------------\n")

        sample_result = LayerThicknessResult(units='nm')

        # Reset intial threshold estimates
        initial_threshold_estimates: Union[List[float], None] = None

        # Lazy load images
        for image, image_path in sample.images_generator():

            print(f"\nProcessing image #{image_tally} : {Path(*image_path.parts[-4:])}")

            save_name = setup_save_location(image_path)
            segmentation_save_path = save_name.with_suffix('.png')  # Save segmentation figure as PNG
            result_save_path = save_name.with_suffix('.json')  # Save this image's LayerThicknessResult as a JSON file

            if all([filename.exists() for filename in (save_name, segmentation_save_path, result_save_path)]):
                # This has already been processed
                response = input("This image has already been processed, overwrite? (y/n): ")
                if response.lower() == 'n':
                    print("Continuing")
                    continue
                else:
                    print("Overwriting")

            if not save_name.exists():
                save_name.symlink_to(image_path)  # Creat a symbolic link to the original datafile in the results area

            smoothed_image = image_smoother(image.data, image.axes_manager['x'].scale, 0.5)
            
            if initial_threshold_estimates == None:
                initial_threshold_estimates = initial_threshold_estimate(smoothed_image)
            refinement_result, refined_thresholds = refine_lines(smoothed_image, initial_threshold_estimates, segmentation_save_path)

            if refinement_result == ImageAction.DISCARD: 
                print(f"Image results discarded")
                continue
            if refinement_result == ImageAction.QUIT: 
                print("Quitting!")
                exit(0)
            if refined_thresholds is None: raise ValueError("This shouldn't be possible???")
            initial_threshold_estimates = refined_thresholds  # Use these as the initial guess for the next image's thresholds

            segmented_image = segment_image(smoothed_image, refined_thresholds)
            layer_thicknesses, pixel_counts = np.unique(segmented_image, return_counts=True)  # Returns (thicknesses, counts)
            layer_thickness_pixel_counts = LayerThicknessResult.from_lists(layer_thicknesses, pixel_counts, refined_thresholds)
            # ^^^ A dict of {0: num px at zero, 1: num px at one, ...}

            layer_thickness_pixel_counts.save(result_save_path)  # Save as JSON in-case it is needed later

            pixel_area = image.axes_manager['x'].scale * image.axes_manager['y'].scale
            if image.axes_manager['x'].units != 'nm':
                print("Unknown units!")
                break
            layer_thickness_pixel_counts.convert_units(pixel_area, 'nm')

            sample_result += layer_thickness_pixel_counts
            image_tally += 1

        sample_result.save(sample_result_save_names[sample])
        print(f"Sample complete, result saved to {sample_result}\n")

    print("Program Complete")
    