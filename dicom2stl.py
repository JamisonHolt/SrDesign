from __future__ import print_function
import SimpleITK as sitk
import sitk2vtk
import vtkutils

def img_to_stl(img):
    # Pad black to the boundaries of the image
    pad = [5, 5, 5]
    img = sitk.ConstantPad(img, pad, pad)
    vtkimg = sitk2vtk.sitk2vtk(img)
    isovalue = 0
    mesh = vtkutils.extractSurface(vtkimg, isovalue)
    connectivityFilter = False
    mesh = vtkutils.cleanMesh(mesh, connectivityFilter)
    smoothIterations = 25
    mesh = vtkutils.smoothMesh(mesh, smoothIterations)
    quad = .90
    mesh = vtkutils.reduceMesh(mesh, quad)

    vtkutils.writeMesh(mesh, "result.stl")
