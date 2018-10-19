import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn import cluster


def show_one(img):
    """
    Display a single 2D image without calling plt.show() to open in the browser

    :param img: The 2D image to be shown
    :return: None
    """
    dpi = 40
    margin = 0.05
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    figsize = (5, 5)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)


def show_all(img, overlay=None, axis='z'):
    """
    Take in all images and display them in the browser on any given axis

    :param img: The image to be displayed
    :param overlay: Any overlay of labels that one might want displayed. Defaults to none
    :param axis: The axis in which to graph each image. Defaults to z
    :return: None
    """
    xlen, ylen, zlen = img.GetSize()
    all_images = []
    all_overlays = []
    if axis == 'z':
        all_images = [img[:, :, z] for z in xrange(zlen)]
        if overlay:
            all_overlays = [overlay[:, :, z] for z in xrange(zlen)]
    elif axis == 'y':
        all_images = [img[:, y, :] for y in xrange(ylen)]
        if overlay:
            all_overlays = [overlay[:, y, :] for y in xrange(ylen)]
    elif axis == 'x':
        all_images = [img[x, :, :] for x in xrange(xlen)]
        if overlay:
            all_overlays = [overlay[x, :, :] for x in xrange(xlen)]
    else:
        raise Exception('invalid axis')

    for i, image in enumerate(all_images):
        if overlay:
            show_one(sitk.LabelOverlay(image, all_overlays[i]))
        else:
            show_one(image)
    plt.show()


def make_empty_img_from_img(img, dimensions=3):
    """
    Take an exising itk image and create a new, empty image from its dimensions

    :param img: The image to find dimensions for
    :param dimensions: The number of dimensions in the image
    :return: The new image
    """
    xlen, ylen, zlen = img.GetSize()
    dupe = img[:, :, :]
    for x in xrange(xlen):
        for y in xrange(ylen):
            if dimensions == 3:
                for z in xrange(zlen):
                    dupe.SetPixel(x, y, z, 0)
            else:
                dupe.SetPixel(x, y, 0)
    return dupe


def read_image(path):
    """
    Read in a list of dcm images in a given directory

    :param path: system path towards the directory
    :return: sitk image with the origin reset to 0, 0, 0
    """
    reader = sitk.ImageSeriesReader()
    dicom_filenames = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_filenames)
    reader.LoadPrivateTagsOn()
    img = reader.Execute()
    img.SetOrigin((0, 0, 0))
    return img


def retrieve_overlap(img1, img2, lbl1=1, lbl2=1):
    """
    Take in two images of labels and return an image with only the overlap of the labels

    :param img1: The first image of labels
    :param img2: The second image of labels
    :param lbl1: The label to retrieve from the first image
    :param lbl2: The label to retrieve from the second image
    :return: A new image of labels where overlap exists
    """
    xlen, ylen, zlen = img1.GetSize()

    # Make sure that our images are equal in size to prevent weird invisible bugs
    xlen2, ylen2, zlen2 = img2.GetSize()
    assert xlen == xlen2 and ylen == ylen2 and zlen == zlen2

    # Copy our image as to not alter the original data
    new_image = img1[:, :, :]
    for z in xrange(zlen):
        for y in xrange(ylen):
            for x in xrange(xlen):
                # Set any bit with overlap to 1, else set it to 0
                overlap = img1.GetPixel(x, y, z) == lbl1 and img2.GetPixel(x, y, z) == lbl2
                if overlap:
                    new_image.SetPixel(x, y, z, 1)
                else:
                    new_image.SetPixel(x, y, z, 0)
    return new_image


def get_df_from_img(img, dimensions=3):
    """
    Create a pandas dataframe from any given image - useful for statistics operations such as clustering

    :param img: The image to be converted into a dataframe
    :param dimensions: The number of dimensions of the image - only supports 2D and 3D images at the moment
    :return: A pandas dataframe containing the x, y, and z coordinates that exist in the image
    """
    if dimensions == 3:
        df_dict = {'x': [], 'y': [], 'z': []}
        xlen, ylen, zlen = img.GetSize()
        for x in xrange(xlen):
            for y in xrange(ylen):
                for z in xrange(zlen):
                    if img.GetPixel(x, y, z):
                        df_dict['x'].append(x)
                        df_dict['y'].append(y)
                        df_dict['z'].append(z)
        df = pd.DataFrame.from_dict(df_dict)
        return df
    elif dimensions == 2:
        df_dict = {'x': [], 'y': []}
        xlen, ylen = img.GetSize()
        for x in xrange(xlen):
            for y in xrange(ylen):
                if img.GetPixel(x, y):
                    df_dict['x'].append(x)
                    df_dict['y'].append(y)
        df = pd.DataFrame.from_dict(df_dict)
        return df
    else:
        raise Exception('Unsupported number of dimensions')


def update_img_from_df(df, image, keep=0, dimensions=3, colname='label', inside_value=1, outside_value=0):
    """
    Take a given dataframe and itk image to be written over and update the image to only contain the labeled coordinates

    :param df: The dataframe to read labels from
    :param image: The image to be overwritten
    :param keep: The label in the dattaframe to keep (since there may be multiple labels, e.g. clustering
    :param dimensions: The number of dimensions in the image
    :param colname: The name of the column containing the labels
    :param inside_value: What to update labeled pixels to
    :param outside_value: What to update unlabeled pixels to
    :return: None
    """
    for index, row in df.iterrows():
        if dimensions == 2:
            x, y, label = (row['x'], row['y'], row[colname])
            if label == keep:
                image.SetPixel(x, y, inside_value)
            else:
                image.SetPixel(x, y, outside_value)
        elif dimensions == 3:
            x, y, z, label = (row['x'], row['y'], row['z'], row[colname])
            if label == keep:
                image.SetPixel(x, y, z, inside_value)
            else:
                image.SetPixel(x, y, z, outside_value)
        else:
            raise Exception('Unsupported number of dimensions')


def dbscan_filter(img, eps, use_z=True):
    df = get_df_from_img(img)
    df_new = df
    if not use_z:
        df_new = df.drop('z', axis=1)
    fit = cluster.DBSCAN(eps=eps).fit(df_new)
    labels = fit.labels_
    df['label'] = pd.Series(labels)
    counts = df['label'].value_counts().to_dict()
    # Remove all non-clusters
    df = df[df.label != -1]
    largest_cluster = max(counts.iterkeys(), key=(lambda key: counts[key]))
    img_filtered = make_empty_img_from_img(img)
    update_img_from_df(df, img_filtered, keep=largest_cluster)
    return img_filtered


def kmeans_segment(img, num_segments=2, use_z=True):
    df = get_df_from_img(img)
    df_new = df
    if not use_z:
        df_new = df.drop('z', axis=1)
    fit = cluster.KMeans(n_clusters=num_segments).fit(df_new)
    labels = fit.labels_
    df['label'] = pd.Series(labels)
    all_images = [make_empty_img_from_img(img) for i in xrange(num_segments)]
    x_max = [0 for i in xrange(num_segments)]
    for index, row in df.iterrows():
        x, y, z, label = (row['x'], row['y'], row['z'], row['label'])
        all_images[label].SetPixel(x, y, z, 1)
        x_max[label] = max((x_max[label], x))
    return all_images, x_max


def count_labels(img):
    xlen, ylen = img.GetSize()
    count = 0
    for x in xrange(xlen):
        for y in xrange(ylen):
            if img.GetPixel(x, y):
                count += 1
    return count


def filter_by_label_count(img, threshold):
    start = 0
    arr = sitk.GetArrayFromImage(img)
    end = len(arr)
    for z in xrange(end):
        img_single = img[:, :, z]
        if count_labels(img_single) < threshold:
            if z == start:
                start += 1
    for z in reversed(xrange(end)):
        img_single = img[:, :, z]
        if count_labels(img_single) < threshold:
            if z == end - 1:
                end -= 1
    return start, end




def main():
    """
    Main function of our program. Executes all of the main steps written in our final paper

    :return: None
    """
    # Directory where the DICOM files are being stored (in this
    input_path = './Inputs/valve'

    # Original image from the filepath
    img_original = read_image(input_path)

    # Image with smoothing applied to reduce noise
    img_smooth = sitk.CurvatureFlow(image1=img_original, timeStep=0.125, numberOfIterations=10)

    # Create labels on our smoothed image for cardiac tissue and tissue with blood
    labels_tissue = sitk.BinaryThreshold(image1=img_smooth, lowerThreshold=325, upperThreshold=470, insideValue=1)
    labels_blood = sitk.BinaryThreshold(image1=img_smooth, lowerThreshold=450, upperThreshold=800, insideValue=1, outsideValue=0)

    # IMPORTANT STEP: essentially, this is the key to our algorithm. By finding the "blood" without cardiac tissue,
    #   and then using binary hole filling with a fairly large radius, we are able to label a lot of the mitral valve
    #   area without labeling too much of the other cardiac tissue. Thus, THIS is what lets us single out the mitral
    #   valve tissue from the rest - all we need is the overlap of the two labels
    labels_tissue_no_holes = sitk.VotingBinaryHoleFilling(image1=labels_tissue, radius=[2] * 3, majorityThreshold=1, backgroundValue=0, foregroundValue=1)
    labels_blood_no_holes = sitk.VotingBinaryHoleFilling(image1=labels_blood, radius=[4] * 3, majorityThreshold=1, backgroundValue=0, foregroundValue=1)
    labels_valve = retrieve_overlap(labels_blood_no_holes, labels_tissue_no_holes)
    labels_valve_no_holes = sitk.VotingBinaryHoleFilling(image1=labels_valve, radius=[2] * 3, majorityThreshold=1, backgroundValue=0, foregroundValue=1)
    labels_valve_no_holes = sitk.VotingBinaryHoleFilling(image1=labels_valve_no_holes, radius=[1] * 3, majorityThreshold=0, backgroundValue=1, foregroundValue=0)

    # Fix intensity scaling on our original smoothed image for pretty diagram purposes
    img_smooth = sitk.Cast(sitk.RescaleIntensity(img_smooth), labels_tissue_no_holes.GetPixelID())

    # Use a density-based clustering algorithm to attempt to remove as much noise as possible
    labels_valve_filtered = dbscan_filter(labels_valve_no_holes, eps=2, use_z=False)
    labels_valve_filtered = dbscan_filter(labels_valve_filtered, eps=4)

    # Find likely start and end points of our image by setting a mininum number of labeled pixels
    start, end = filter_by_label_count(labels_valve_filtered, 10)
    img_smooth = img_smooth[:, :, start:end]
    labels_valve_filtered = labels_valve_filtered[:, :, start:end]

    # Remove all values distant from the center of our starting location by taking advantage of kmeans
    df = get_df_from_img(labels_valve_filtered[:, :, 0], dimensions=2)
    x_mid = df['x'].mean()
    y_mid = df['y'].mean()
    df = get_df_from_img(labels_valve_filtered)
    distance_df = df.drop('z', axis=1)
    distance_df['x_dist'] = abs(distance_df['x'] - x_mid)
    distance_df['y_dist'] = abs(distance_df['y'] - y_mid)
    fit = cluster.KMeans(n_clusters=2).fit(distance_df.drop(['x', 'y'], axis=1))
    labels = fit.labels_
    df['label'] = pd.Series(labels)
    counts = df['label'].value_counts().to_dict()
    largest_cluster = max(counts.iterkeys(), key=(lambda key: counts[key]))
    update_img_from_df(df, labels_valve_filtered, keep=largest_cluster)

    # Find likely start and end points of our image by setting a mininum number of labeled pixels
    start, end = filter_by_label_count(labels_valve_filtered, 10)
    img_smooth = img_smooth[:, :, start:end]
    labels_valve_filtered = labels_valve_filtered[:, :, start:end]

    # Use a segmentation-based clustering algorithm to attempt to find each valve
    label_segments, x_max = kmeans_segment(labels_valve_filtered, use_z=False)

    left, right = (label_segments[0], label_segments[1])
    if x_max[0] > x_max[1]:
        left, right = right, left

    # Finally, we can simply take the furthest point from the likely start/end points in order to get our annulus
    # this can be done by every z value
    left_points = {'x': [], 'y': [], 'z': []}
    right_points = {'x': [], 'y': [], 'z': []}
    zlen = len(sitk.GetArrayFromImage(left))
    for z in xrange(zlen):
        left_df = get_df_from_img(left[:, :, z], dimensions=2)
        if len(left_df['y']) > 0:
            index = left_df['y'].idxmin()
            row = left_df.iloc[index]
            left_points['x'].append(int(row['x']))
            left_points['y'].append(int(row['y']))
            left_points['z'].append(z)

        right_df = get_df_from_img(right[:, :, z], dimensions=2)
        if len(right_df['x']) > 0:
            index = right_df['x'].idxmax()
            row = right_df.iloc[index]
            right_points['x'].append(int(row['x']))
            right_points['y'].append(int(row['y']))
            right_points['z'].append(z)

    # These both represent the coordinates of our annulus ring. A simple spline can be used for interpolation between
    #   points
    final_left = pd.DataFrame.from_dict(left_points)
    final_right = pd.DataFrame.from_dict(right_points)
    print('Coordinates for one side of the ring')
    print(final_left)
    print('\n\nCoordinates for the other side of the ring')
    print(final_right)

    final_image = make_empty_img_from_img(left)
    x = left_points['x'] + right_points['x']
    y = left_points['y'] + right_points['y']
    z = left_points['z'] + right_points['z']
    for x, y, z in zip(x, y, z):
        final_image.SetPixel(x, y, z, 1)

    show_all(img_smooth, final_image)


if __name__ == '__main__':
    main()