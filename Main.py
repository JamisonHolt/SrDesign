import dicom2stl
import numpy as np
import time
import SimpleITK as sitk
import matplotlib.pyplot as plt
import sys


def sitk_show(img, title=None, margin=0.05, dpi=40):
    """
    This function uses matplotlib.pyplot to quickly visualize a 2D sitk.Image object under the img parameter.

    :param img:
    :param title:
    :param margin:
    :param dpi:
    :return:
    """

    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    figsize = (5, 5)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)


def writeImage(img, reader, path):
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    direction = img.GetDirection()

    series_tag_values = [
        ("0010|0010", "Test1"),  # Patient Name
        ("0010|0020", "Test1"),  # Patient ID
        ("0010|0030", "20060101"),  # Patient Birth Date
        ("0020|000D", "1.2.826.0.1.3680043.2.1125.1.94286748084135693443502286530644958"),  # Study Instance UID, for machine consumption
        ("0020|0010", "SLICER100011"),  # Study ID, for human consumption
        ("0008|0020", "20060101"),  # Study Date
        ("0008|0030", "010100.000000"),  # Study Time
        ("0008|0050", "1"),  # Accession Number
        ("0008|0060", "CT"),  # Modality
        ("0008|0031", modification_time),  # Series Time
        ("0008|0021", modification_date),  # Series Date
        ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
        ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
        # Series Instance UID
        ("0020|0037",
        '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
            direction[1], direction[4], direction[7])))),
        ("0008|103e", "Processed Image")]  # Series Description
    for i in range(img.GetDepth()):
        image_slice = img[:,:,i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str,img.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i)) # Instance Number

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(path + 'output' + str(i)+'.dcm')
        writer.Execute(image_slice)

def getSafePoints(grid, radius=8, threshold=15):
    heatMap = np.zeros((len(grid[0]), len(grid[0][0])))
    safePoints = np.copy(heatMap);
    for z in range(len(grid)):
        for y in range(len(grid[z])):
            for x in range(len(grid[z][y])):
                heatMap[y][x] += grid[z][y][x]
                # if z == 0 or y == 0 or x == 0 or z == (len(grid) - 1) or y == (len(grid[z]) - 1) or x == (len(grid[z][y]) - 1):
                #     grid[z][y][x] = 0
                # elif not(grid[z+1][y][x] or grid[z][y+1][x] or grid[z][y-1][x] or grid[z][y][x+1] or grid[z][y][x-1]):
                #     grid[z][y][x] = 0
    for y in range(len(heatMap)):
        for x in range(len(heatMap[y])):
            if heatMap[y][x] >= threshold:
                lowX = x - radius if x >= radius else 0
                lowY = y - radius if y >= radius else 0
                highX = x + radius if x < len(heatMap[y]) else len(heatMap[y]) - 1
                highY = y + radius if y < len(heatMap) else len(heatMap) - 1
                safePoints[lowY:highY, lowX:highX] = 1
    return safePoints


def labelIslands(grid):
    visited = np.zeros((len(grid[0]), len(grid[0][0])))
    def recur(plane, loc, label):
        row, col = loc
        if row < 0 or col < 0 or row == len(plane) or col == len(plane[0]):
            return
        if (visited[row][col]):
            return
        visited[row][col] = 1
        if plane[row][col] == 1:
            plane[row][col] = label
            recur(plane, (row - 1, col), label)
            recur(plane, (row + 1, col), label)
            recur(plane, (row, col - 1), label)
            recur(plane, (row, col + 1), label)
    label = 2
    gridCopy = np.copy(grid)
    heightMap = {}
    for z, plane in enumerate(gridCopy):
        for y in range(len(plane)):
            for x in range(len(plane[y])):
                if plane[y][x] == 1:
                    recur(plane, (y, x), label)
                    heightMap[label] = z
                    label += 1
        visited[:, :] = 0
    return gridCopy, heightMap


def getHouse(top, bottom, columns, connectedSets):
    house = set([top, bottom])
    for colIndex in connectedSets:
        if colIndex in columns and bottom in columns[colIndex]:
            house.add(colIndex)
    return house

def buildHouses(grid, heightMap):
    islands = set()
    columns = {}
    houses = []
    for y in range(len(grid[0])):
        for x in range(len(grid[0][y])):
            label = grid[0][y][x]
            if label:
                islands.add(label)
    for z in range(1, len(grid)):
    # for z in range(1, 5):
        isleReferenceCount = {}
        newColumns = {}
        newIslands = set()
        for y in range(len(grid[z])):
            for x in range(len(grid[z][y])):
                label = grid[z][y][x]
                below = grid[z-1][y][x]
                # All labels are islands for the purposes of this code, even if also columns
                if label:
                    newIslands.add(label)
                # If label is a column to an island below
                if label and below:
                    # Check if previously added to columns
                    if label not in newColumns:
                        newColumns[label] = set()
                    # Check if below is a column
                    if below in columns:
                        newColumns[label] = newColumns[label].union(columns[below])
                        newColumns[label].add(below)
                    # Case that below is simply an island
                    elif below in islands and below not in newColumns[label]:
                        # Add column to our dictionary
                        newColumns[label].add(below)
                        # keep count of times island referenced - we only care about (2+)-referenced islands
                        if below not in isleReferenceCount:
                            isleReferenceCount[below] = 0
                        isleReferenceCount[below] += 1
                        # filter and merge newColumns with old Columns

        for isle, count in isleReferenceCount.items():
            # Remove all islands with fewer than 2 columns bc they aren't contenders
            for islandTop, islandsBot in newColumns.items():
                if count < 2 and isle in islandsBot:
                    islandsBot.remove(isle)
                    # Check if any purpose to
                    if len(islandsBot) == 0:
                        newColumns.pop(islandTop)

        ceilingToFloor = {}

        for islandTop, islandsBot in newColumns.items():
            # Find all ceilings connected to a floor by at least 2 separate columns
            if len(islandsBot) > 2:
                childSets = [columns[i] if i in columns else set() for i in islandsBot]
                numIntersections = {}
                for childSet in childSets:
                    # find the floor connected to this ceiling
                    for island in childSet:
                        if island not in numIntersections:
                            numIntersections[island] = 0
                        numIntersections[island] += 1
                        if numIntersections[island] > 1:
                            ceilingToFloor[islandTop] = island
                house = getHouse(islandTop, island, columns, islandsBot)
                if min(house) != 1:
                    houses.append((heightMap[max(house)] - heightMap[min(house)], house))
        islands = newIslands
        columns.update(newColumns)
    return houses

def main():
    # Directory where the DICOM files are being stored (in this
    pathDicom = './Inputs/valve'
    pathOutput = './Outputs/valve/'

    # int labels to assign to the segmented white and gray matter.
    # These need to be different integers but their values themselves
    # don't matter
    labelWhiteMatter = 1

    reader = sitk.ImageSeriesReader()
    seriesID = reader.GetGDCMSeriesIDs(pathDicom)[0]
    series_file_names = reader.GetGDCMSeriesFileNames(pathDicom, seriesID)
    reader.SetFileNames(series_file_names)
    filenamesDICOM = reader.GetGDCMSeriesFileNames(pathDicom)
    reader.SetFileNames(filenamesDICOM)
    reader.LoadPrivateTagsOn()
    imgOriginal = reader.Execute()

    imgOriginal.SetOrigin((0, 0, 0))

    imgSmooth = sitk.CurvatureFlow(image1=imgOriginal, timeStep=0.125, numberOfIterations=10)

    edgeSeeds = [(28, 31, 37)]
    imgWhiteMatter = sitk.ConnectedThreshold(image1=imgSmooth, seedList=edgeSeeds, lower=200, upper=470, replaceValue=labelWhiteMatter)
    bloodSeeds = [(23, 35, 23)]
    imgBlood = sitk.BinaryThreshold(image1=imgSmooth, lowerThreshold=300, upperThreshold=800, insideValue=1, outsideValue=0)
    # imgBlood = sitk.ConnectedThreshold(image1=imgSmooth, seedList=bloodSeeds, lower=257, upper=1000, replaceValue=labelWhiteMatter)
    imgSmoothInt = sitk.Cast(sitk.RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())
    imgWhiteMatterNoHoles = sitk.VotingBinaryHoleFilling(image1=imgWhiteMatter, radius=[2] * 3, majorityThreshold=1, backgroundValue=0, foregroundValue=labelWhiteMatter)
    imgMasked = sitk.Mask(imgSmoothInt, imgWhiteMatterNoHoles)
    imgCoords = sitk.GetArrayFromImage(imgWhiteMatterNoHoles)


    for z in range(len(imgCoords)):
        for y in range(len(imgCoords[z])):
            for x in range(len(imgCoords[z][y])):
                overlap = imgWhiteMatterNoHoles.GetPixel(x, y, z) == 1 and imgBlood.GetPixel(x, y, z) == 1
                if not overlap:
                    imgBlood.SetPixel(x, y, z, 0)

    imgCoords = sitk.GetArrayFromImage(imgBlood)

    islands, heightMap = labelIslands(imgCoords)
    houses = buildHouses(islands, heightMap)
    houses.sort()
    height, house = houses[-1]
    combinedHouse = set()
    for house in houses:
        if house[0] >= 10:
            combinedHouse.update(house[1])
    for z in range(len(islands)):
        for y in range(len(islands[z])):
            for x in range(len(islands[z][y])):
                if islands[z][y][x] in combinedHouse:
                    islands[z][y][x] = 1
                    imgBlood.SetPixel(x, y, z, 1)
                else:
                    islands[z][y][x] = 0
                    imgBlood.SetPixel(x, y, z, 0)

    safePoints = getSafePoints(islands)



    for i in range(imgOriginal.GetHeight()):
        imgWhiteMatterSingle = imgWhiteMatter[:, :, i]
        imgSmoothIntSingle = imgSmoothInt[:, :, i]
        imgWhiteMatterNoHolesSingle = imgWhiteMatterNoHoles[:, :, i]
        imgBloodSingle = imgBlood[:, :, i]
        for y in range(len(safePoints)):
            for x in range(len(safePoints[y])):
                if not safePoints[y][x]:
                    imgBloodSingle.SetPixel(x, y, 0)
        imgMaskedSingle = imgMasked[:, :, i]
        # Use 'LabelOverlay' to overlay 'imgSmooth' and 'imgWhiteMatterSingle'
        # sitk_show(sitk.LabelOverlay(imgSmoothIntSingle, imgWhiteMatterNoHolesSingle))
        # sitk_show(sitk.LabelOverlay(imgSmoothIntSingle, imgBloodSingle))
        sitk_show(sitk.LabelOverlay(imgSmoothIntSingle, imgBloodSingle))
        # sitk_show(imgBloodSingle)
        # sitk_show(imgSmoothIntSingle)
        # sitk_show(imgWhiteMatterNoHolesSingle)

    plt.show()

if __name__ == '__main__':
    main()