import dicom2stl
import SimpleITK
import matplotlib.pyplot as plt


def sitk_show(img, title=None, margin=0.05, dpi=40):
    """
    This function uses matplotlib.pyplot to quickly visualize a 2D SimpleITK.Image object under the img parameter.

    :param img:
    :param title:
    :param margin:
    :param dpi:
    :return:
    """

    nda = SimpleITK.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    # figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    figsize = (5, 5)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)


def main():
    # Directory where the DICOM files are being stored (in this
    pathDicom = './Inputs/valve'

    # Z slice of the DICOM files to process. In the interest of
    # simplicity, segmentation will be limited to a single 2D
    # image but all processes are entirely applicable to the 3D image
    idxSlice = 25

    # int labels to assign to the segmented white and gray matter.
    # These need to be different integers but their values themselves
    # don't matter
    labelWhiteMatter = 1
    labelGrayMatter = 2

    reader = SimpleITK.ImageSeriesReader()
    filenamesDICOM = reader.GetGDCMSeriesFileNames(pathDicom)
    reader.SetFileNames(filenamesDICOM)
    imgOriginal = reader.Execute()

    print(imgOriginal.GetDimension())
    print(imgOriginal.GetDepth())
    print(imgOriginal.GetHeight())
    print(imgOriginal.GetDirection())
    print(imgOriginal.GetSpacing())
    print(imgOriginal.GetOrigin())
    imgOriginal.SetOrigin((0, 0, 0))
    print(imgOriginal.GetOrigin())
    print(type(imgOriginal) == SimpleITK.SimpleITK.Image)

    # imgOriginal = imgOriginal[:, :, idxSlice]
    imgSmooth = SimpleITK.CurvatureFlow(image1=imgOriginal,
                                        timeStep=0.125,
                                        numberOfIterations=10)

    # blurFilter = SimpleITK.CurvatureFlowImageFilter()
    # blurFilter.SetNumberOfIterations(5)
    # blurFilter.SetTimeStep(0.125)
    # imgSmooth = blurFilter.Execute(imgOriginal)

    # sitk_show(imgSmooth)

    lstSeeds = [(28, 31, 37), (22, 12, 22), (50, 40, 22), (28, 31, 37)]

    imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth,
                                                  seedList=lstSeeds,
                                                  # May want to adjust lower
                                                  lower=300,
                                                  upper=459,
                                                  replaceValue=labelWhiteMatter)

    # Rescale 'imgSmooth' and cast it to an integer type to match that of 'imgWhiteMatter'
    imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())


    imgWhiteMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgWhiteMatter,
                                                              radius=[2] * 3,
                                                              majorityThreshold=1,
                                                              backgroundValue=0,
                                                              foregroundValue=labelWhiteMatter)

    dicom2stl.img_to_stl(imgWhiteMatterNoHoles)
    # for i in range(imgOriginal.GetDepth()):
    #     imgWhiteMatterSingle = imgWhiteMatter[:, :, i]
    #     imgSmoothIntSingle = imgSmoothInt[:, :, i]
    #     imgWhiteMatterNoHolesSingle = imgWhiteMatterNoHoles[:, :, i]
    #     # Use 'LabelOverlay' to overlay 'imgSmooth' and 'imgWhiteMatterSingle'
    #     print(type(imgWhiteMatterNoHoles))
    #     sitk_show(SimpleITK.LabelOverlay(imgSmoothIntSingle, imgWhiteMatterNoHolesSingle))
    #     # sitk_show(imgSmoothIntSingle)
    # plt.show()


if __name__ == '__main__':
    main()