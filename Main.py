import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from glob import glob
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import scipy.ndimage
# from skimage import morphology
# from skimage import measure
# from skimage.transform import resize
# from sklearn.cluster import KMeans
# from plotly import __version__
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# from plotly.tools import FigureFactory as FF
# from plotly.graph_objs import *


def load_scan(path):
    """
    Loop over the image files and store everything into a list

    :param path: String of the input to be taken in
    :return: list of DICOM filedataset types encompassing the scan
    """
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(scans):
    """
    Read in a list of scans and parse the pixels from thems

    :param scans: list of scans to be taken in
    :return: np.array of the image pixels
    """
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def pre_process(data_path, output_path):
    id = 0
    patient = load_scan(data_path)
    imgs = get_pixels_hu(patient)
    np.save(output_path + "fullimages.npy", imgs)


def main():
    data_path = "./Inputs/full/"
    output_path = "./Outputs/full/"
    g = glob(data_path + '/*.dcm')

    file_used = output_path + "fullimages.npy"
    imgs_to_process = np.load(file_used).astype(np.float64)

    plt.hist(imgs_to_process.flatten(), bins=50, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == '__main__':
    main()