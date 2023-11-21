# -*- coding: utf-8 -*-

"""
@author: Chenfei
@contact:chenfei.ye@foxmail.com
@file: object_visualization.py
@time: 2021/08/09
"""

import subprocess
import vtk
import traceback
import math
import os


def MarchingCubes(image,threshold): 
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.ComputeNormalsOn()
    mc.ComputeGradientsOn()
    mc.SetValue(0, threshold)
    mc.Update()
    return mc.GetOutput()


def Smooth_vtk_data(stl):
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(stl)
    smoothFilter.SetNumberOfIterations(5)
    smoothFilter.SetRelaxationFactor(0.1)
    smoothFilter.FeatureEdgeSmoothingOff()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()
    return smoothFilter.GetOutput()


def tck2vtk(intckfile, outvtkfile, reference_nii):
    """
    convert tck to vtk using MRtrix
    intckfile: filepath of single tck file 
    outvtkfile: filepath of single vtk file 
    reference_nii: filepath of reference nifti image
    """
    subprocess.run(['tckconvert -scanner2image ' + reference_nii + ' ' + intckfile + ' ' + outvtkfile], check=True, shell=True)


def tck2vtk_batch(indir, outdir, reference_nii, bundle_ls):
    """
    batch conversion from tck to vtk
    indir：input directory of tck files
    outdir: output directory of vtk files
    bundle_ls: list of fiber bundles
    """
    for i in bundle_ls:
        intckfile = os.path.join(indir, i + '.tck')
        outvtkfile = os.path.join(outdir, i + '.vtk')
        tck2vtk(intckfile, outvtkfile, reference_nii)


def vtk2vtp(invtkfile, outvtpfile, addcolor=True, binary=False):
    """
    convert vtk ro vtp with color
    developed by Alfred
    invtkfile: filepath of single vtk file 
    outvtpfile: filepath of single vtp file 
    """
    try:
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(invtkfile)
        reader.Update()
        polyDataOutput = reader.GetOutput()
        if addcolor:
            # load all streamlines
            unprocessedStreamlines = polyDataOutput.GetLines()
            unprocessedStreamlines.InitTraversal()
            fiberPointVectors = {}
            colorVector = [0,0,0]
            idList = vtk.vtkIdList()
            # for each streamline
            while unprocessedStreamlines.GetNextCell(idList):
                # for id of each point
                total = idList.GetNumberOfIds()
                # set color
                for i in range(0,total -1):
                    currentPointId = idList.GetId(i)
                    nextPointId = idList.GetId(i+1)
                    currentPointCoord = [0,0,0]
                    nextPointCoord = [0,0,0]
                    polyDataOutput.GetPoint(currentPointId, currentPointCoord)
                    polyDataOutput.GetPoint(nextPointId, nextPointCoord)
                    distance = math.sqrt(pow(nextPointCoord[0] - currentPointCoord[0], 2) + pow(nextPointCoord[1] - currentPointCoord[1], 2) + pow(nextPointCoord[2] - currentPointCoord[2], 2))
                    colorVector = [(int)(255 * abs((nextPointCoord[0] - currentPointCoord[0]) / distance)),(int)(255 * abs((nextPointCoord[1] - currentPointCoord[1]) / distance)),(int)(255 * abs((nextPointCoord[2] - currentPointCoord[2]) / distance))]
                    fiberPointVectors[currentPointId] = colorVector
                    if i == total - 2:
                        fiberPointVectors[nextPointId] = colorVector
            color = vtk.vtkUnsignedCharArray()
            color.SetName("colors")
            color.SetNumberOfComponents(3)
            pointColor = [0,0,0]
            pointQuantity = reader.GetOutput().GetNumberOfPoints()
            # set color for each point
            for i in range(0,pointQuantity):
                if i in fiberPointVectors.keys():
                    pointColor = fiberPointVectors[i]
                else:
                    pointColor = [255,255,255]
                color.InsertNextTypedTuple(pointColor)
            reader.GetOutput().GetPointData().SetScalars(color)
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(outvtpfile)
            if binary:
                writer.SetFileTypeToBinary()
            writer.SetInputData(reader.GetOutput())
            writer.Update()
        else:
            normalGenerator = vtk.vtkPolyDataNormals()
            normalGenerator.SetInputConnection(reader.GetOutputPort())
            normalGenerator.ComputePointNormalsOn()
            normalGenerator.ComputeCellNormalsOn()
            normalGenerator.Update()
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(outvtpfile)
            if binary:
                writer.SetFileTypeToBinary()
            writer.SetInputData(normalGenerator.GetOutput())
            writer.Update()
    except Exception as e:        
        print(traceback.format_exc())
        print('exception happened..: %s' % e)


def vtk2vtp_batch(indir, outdir, bundle_ls):
    """
    batch conversion from vtk to vtp
    indir：input directory of vtk files
    outdir: output directory of vtp files
    bundle_ls: list of fiber bundles
    """
    for i in bundle_ls:
        intckfile = os.path.join(indir, i + '.vtk')
        outvtkfile = os.path.join(outdir, i + '.vtp')
        vtk2vtp(intckfile, outvtkfile)