import numpy as np
import pandas as pd
import time
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn import cluster

def sitk_show(img, margin=0.05, dpi=40):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    figsize = (5, 5)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)


def show_img(img, overlay=None):
    for z in range(img.GetHeight()):
        img_single = img[:, :, z]
        if overlay:
            overlay_single = overlay[:, :, z]
            sitk_show(sitk.LabelOverlay(img_single, overlay_single))
        else:
            sitk_show(img_single)

def read_image(path):
    # Create a DICOM reader and load in all images
    reader = sitk.ImageSeriesReader()
    series_id = reader.GetGDCMSeriesIDs(path)[0]
    series_file_names = reader.GetGDCMSeriesFileNames(path, series_id)
    reader.SetFileNames(series_file_names)
    reader.LoadPrivateTagsOn()
    img = reader.Execute()

    # Some files have arbitrary origins - normalize this
    img.SetOrigin((0, 0, 0))

    return img


def main():
    # Read in our test file
    input_path = './Inputs/valve'
    img_original = read_image(input_path)

    # Save our image dimensions for future use
    xlen, ylen, zlen = img_original.GetSize()

    # Smooth image to remove abundance of pixels
    img_smooth = sitk.CurvatureFlow(image1=img_original, timeStep=0.125, numberOfIterations=10)
    imgWhiteMatter = sitk.ConnectedThreshold(image1=img_smooth, seedList=[(28, 31, 37)], lower=200, upper=470, replaceValue=1)

    img_smooth = sitk.Cast(sitk.RescaleIntensity(img_smooth), imgWhiteMatter.GetPixelID())

    # Segment image to tissue and blood samples
    # label_tissue = sitk.BinaryThreshold(image1=img_smooth, lowerThreshold=200, upperThreshold=470, insideValue=1)
    label_blood = sitk.BinaryThreshold(image1=img_smooth, lowerThreshold=300, upperThreshold=800, insideValue=1)
    show_img(img_smooth, overlay=label_blood)

    for y in range(img_original.GetHeight()):
        imgSmoothIntSingle = img_smooth[:, y, :]
        imgBloodSingle = label_blood[:, y, :]
        sitk_show(sitk.LabelOverlay(imgSmoothIntSingle, imgBloodSingle))

if __name__ == '__main__':
    main()


