# convert from .off and .txt for .h5

import os
import vtk
import glob
import h5py
import trimesh
import numpy as np
from tqdm import tqdm

from vtk.util.numpy_support import vtk_to_numpy

import pdb


data_dir = '../hdf5_data'
chunk_size = 200

RELAXATION_FACTOR = .2


if __name__ == '__main__':

    # # TRAIN ==================================================
    # train_dir = os.path.join(data_dir, 'training')
    # tr_files = sorted(glob.glob(os.path.join(train_dir, '*', 'triangulatedSurf.off')))
    #
    # idx = 0
    # # data, normal, curvature, label, start, end, pid = [], [], [], [], [], [], []
    # data, curvatureM, curvatureG, curvatureMAX, curvatureMIN, label, start, end, pid = \
    #     [], [], [], [], [], [], [], [], []
    # for off_file in tqdm(tr_files):
    #     # read vertices in .off file
    #     mesh = trimesh.load(off_file, process=False, maintain_order=True)
    #     ply_file = off_file.replace('off', 'ply')
    #     # mesh.export(ply_file)
    #
    #     # vertices = mesh.vertices
    #
    #     # Laplacian smoothing
    #     source = vtk.vtkPLYReader()
    #     source.SetFileName(ply_file)
    #     source.Update()
    #
    #     smooth = vtk.vtkSmoothPolyDataFilter()
    #     smooth.SetInputConnection(source.GetOutputPort())
    #     smooth.SetNumberOfIterations(200)
    #     smooth.SetGenerateErrorVectors(True)
    #     smooth.SetRelaxationFactor(RELAXATION_FACTOR)
    #     smooth.Update()
    #
    #     # # compute vertex (unit) normal
    #     # normalV = vtk.vtkPolyDataNormals()
    #     # normalV.SetInputConnection(smooth.GetOutputPort())
    #     # normalV.Update()
    #     # normals = vtk_to_numpy(normalV.GetOutput().GetPoints().GetData())
    #     # normals = normals / np.sqrt(np.sum(normals ** 2, axis=1, keepdims=True))
    #     #
    #     # if normals.shape[0] != mesh.vertices.shape[0]:
    #     #     pdb.set_trace()
    #
    #     # compute mean curvature
    #     _curvatureM = vtk.vtkCurvatures()
    #     _curvatureM.SetCurvatureTypeToMean()
    #     _curvatureM.SetInputData(smooth.GetOutput())
    #     _curvatureM.Update()
    #     mean_curvature = vtk_to_numpy(_curvatureM.GetOutput().GetPointData().GetScalars())
    #
    #     # compute gaussian curvature
    #     _curvatureG = vtk.vtkCurvatures()
    #     _curvatureG.SetCurvatureTypeToGaussian()
    #     _curvatureG.SetInputData(smooth.GetOutput())
    #     _curvatureG.Update()
    #     gaussian_curvature = vtk_to_numpy(_curvatureG.GetOutput().GetPointData().GetScalars())
    #
    #     # compute maximum curvature
    #     _curvatureMAX = vtk.vtkCurvatures()
    #     _curvatureMAX.SetCurvatureTypeToMaximum()
    #     _curvatureMAX.SetInputData(smooth.GetOutput())
    #     _curvatureMAX.Update()
    #     maximum_curvature = vtk_to_numpy(_curvatureMAX.GetOutput().GetPointData().GetScalars())
    #
    #     # compute minimum curvature
    #     _curvatureMIN = vtk.vtkCurvatures()
    #     _curvatureMIN.SetCurvatureTypeToMinimum()
    #     _curvatureMIN.SetInputData(smooth.GetOutput())
    #     _curvatureMIN.Update()
    #     minimum_curvature = vtk_to_numpy(_curvatureMIN.GetOutput().GetPointData().GetScalars())
    #
    #     vertices = vtk_to_numpy(smooth.GetOutput().GetPoints().GetData())
    #
    #     # read labels in .txt file
    #     map_file = os.path.join(os.path.dirname(off_file), 'vertexMap.txt')
    #     with open(map_file, 'r') as f:
    #         seg = np.array([int(x) for x in f.readlines()])
    #
    #     # put data together
    #     data.append(vertices)
    #     # normal.append(normals)
    #     curvatureM.append(mean_curvature)
    #     curvatureG.append(gaussian_curvature)
    #     curvatureMAX.append(maximum_curvature)
    #     curvatureMIN.append(minimum_curvature)
    #     label.append(seg)
    #     start.append(0 if len(end) == 0 else end[-1])
    #     end.append(start[-1] + vertices.shape[0])  # exclude
    #     pid.append(int(os.path.dirname(off_file).split('/')[-1]))
    #
    #     idx += 1
    #     print('{} --- done --- start: {} --- end: {}'.format(idx, start[-1], end[-1]))
    #
    #     if (idx + 1) % chunk_size == 0:
    #         # save to .h5 file
    #         with h5py.File(os.path.join(data_dir, 'training_{}.h5'.format(idx // chunk_size)), 'w') as hf:
    #             hf.create_dataset('data', data=np.concatenate(data).astype(float))
    #             # hf.create_dataset('normal', data=np.concatenate(normal).astype(float))
    #             hf.create_dataset('curvatureM', data=np.concatenate(curvatureM).astype(float))
    #             hf.create_dataset('curvatureG', data=np.concatenate(curvatureG).astype(float))
    #             hf.create_dataset('curvatureMAX', data=np.concatenate(curvatureMAX).astype(float))
    #             hf.create_dataset('curvatureMIN', data=np.concatenate(curvatureMIN).astype(float))
    #             hf.create_dataset('label', data=np.concatenate(label).astype(int))
    #             hf.create_dataset('start', data=np.array(start).astype(int))
    #             hf.create_dataset('end', data=np.array(end).astype(int))
    #             hf.create_dataset('pid', data=np.array(pid).astype(int))
    #
    #         # data, normal, curvature, label, start, end, pid = [], [], [], [], [], [], []
    #         data, curvatureM, curvatureG, curvatureMAX, curvatureMIN, label, start, end, pid = \
    #             [], [], [], [], [], [], [], [], []
    #
    # # save the rest to .h5 file
    # with h5py.File(os.path.join(data_dir, 'training_{}.h5'.format(idx // chunk_size)), 'w') as hf:
    #     hf.create_dataset('data', data=np.concatenate(data).astype(float))
    #     # hf.create_dataset('normal', data=np.concatenate(normal).astype(float))
    #     hf.create_dataset('curvatureM', data=np.concatenate(curvatureM).astype(float))
    #     hf.create_dataset('curvatureG', data=np.concatenate(curvatureG).astype(float))
    #     hf.create_dataset('curvatureMAX', data=np.concatenate(curvatureMAX).astype(float))
    #     hf.create_dataset('curvatureMIN', data=np.concatenate(curvatureMIN).astype(float))
    #     hf.create_dataset('label', data=np.concatenate(label).astype(int))
    #     hf.create_dataset('start', data=np.array(start).astype(int))
    #     hf.create_dataset('end', data=np.array(end).astype(int))
    #     hf.create_dataset('pid', data=np.array(pid).astype(int))


    # TEST ==================================================
    test_dir = os.path.join(data_dir, 'test')
    te_files = sorted(glob.glob(os.path.join(test_dir, '*', 'triangulatedSurf.off')))

    idx = 0
    # data, normal, curvature, label, start, end, pid = [], [], [], [], [], [], []
    data, curvatureM, curvatureG, curvatureMAX, curvatureMIN, label, start, end, pid = \
        [], [], [], [], [], [], [], [], []
    for off_file in tqdm(te_files):
        # read vertices in .off file
        mesh = trimesh.load(off_file, process=False, maintain_order=True)
        ply_file = off_file.replace('off', 'ply')
        # mesh.export(ply_file)

        # vertices = mesh.vertices

        # Laplacian smoothing
        source = vtk.vtkPLYReader()
        source.SetFileName(ply_file)
        source.Update()

        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputConnection(source.GetOutputPort())
        smooth.SetNumberOfIterations(200)
        smooth.SetGenerateErrorVectors(True)
        smooth.SetRelaxationFactor(RELAXATION_FACTOR)
        smooth.Update()

        # # compute vertex (unit) normal
        # normalV = vtk.vtkPolyDataNormals()
        # normalV.SetInputConnection(smooth.GetOutputPort())
        # normalV.Update()
        # normals = vtk_to_numpy(normalV.GetOutput().GetPoints().GetData())
        # normals = normals / np.sqrt(np.sum(normals ** 2, axis=1, keepdims=True))

        # compute mean curvature
        _curvatureM = vtk.vtkCurvatures()
        _curvatureM.SetCurvatureTypeToMean()
        _curvatureM.SetInputData(smooth.GetOutput())
        _curvatureM.Update()
        mean_curvature = vtk_to_numpy(_curvatureM.GetOutput().GetPointData().GetScalars())

        # compute gaussian curvature
        _curvatureG = vtk.vtkCurvatures()
        _curvatureG.SetCurvatureTypeToGaussian()
        _curvatureG.SetInputData(smooth.GetOutput())
        _curvatureG.Update()
        gaussian_curvature = vtk_to_numpy(_curvatureG.GetOutput().GetPointData().GetScalars())

        # compute maximum curvature
        _curvatureMAX = vtk.vtkCurvatures()
        _curvatureMAX.SetCurvatureTypeToMaximum()
        _curvatureMAX.SetInputData(smooth.GetOutput())
        _curvatureMAX.Update()
        maximum_curvature = vtk_to_numpy(_curvatureMAX.GetOutput().GetPointData().GetScalars())

        # compute minimum curvature
        _curvatureMIN = vtk.vtkCurvatures()
        _curvatureMIN.SetCurvatureTypeToMinimum()
        _curvatureMIN.SetInputData(smooth.GetOutput())
        _curvatureMIN.Update()
        minimum_curvature = vtk_to_numpy(_curvatureMIN.GetOutput().GetPointData().GetScalars())

        vertices = vtk_to_numpy(smooth.GetOutput().GetPoints().GetData())

        # put data together
        data.append(vertices)
        # normal.append(normals)
        curvatureM.append(mean_curvature)
        curvatureG.append(gaussian_curvature)
        curvatureMAX.append(maximum_curvature)
        curvatureMIN.append(minimum_curvature)
        start.append(0 if len(end) == 0 else end[-1])
        end.append(start[-1] + vertices.shape[0])  # exclude
        pid.append(int(os.path.dirname(off_file).split('/')[-1]))

        idx += 1
        print('{} --- done.'.format(idx))

        if (idx + 1) % chunk_size == 0:
            # save to .h5 file
            with h5py.File(os.path.join(data_dir, 'test_{}.h5'.format(idx // chunk_size)), 'w') as hf:
                hf.create_dataset('data', data=np.concatenate(data).astype(float))
                # hf.create_dataset('normal', data=np.concatenate(normal).astype(float))
                hf.create_dataset('curvatureM', data=np.concatenate(curvatureM).astype(float))
                hf.create_dataset('curvatureG', data=np.concatenate(curvatureG).astype(float))
                hf.create_dataset('curvatureMAX', data=np.concatenate(curvatureMAX).astype(float))
                hf.create_dataset('curvatureMIN', data=np.concatenate(curvatureMIN).astype(float))
                hf.create_dataset('start', data=np.array(start).astype(int))
                hf.create_dataset('end', data=np.array(end).astype(int))
                hf.create_dataset('pid', data=np.array(pid).astype(int))

            # data, normal, curvature, label, start, end, pid = [], [], [], [], [], [], []
            data, curvatureM, curvatureG, curvatureMAX, curvatureMIN, label, start, end, pid = \
                [], [], [], [], [], [], [], [], []

    # save the rest to .h5 file
    with h5py.File(os.path.join(data_dir, 'test_{}.h5'.format(idx // chunk_size)), 'w') as hf:
        hf.create_dataset('data', data=np.concatenate(data).astype(float))
        # hf.create_dataset('normal', data=np.concatenate(normal).astype(float))
        hf.create_dataset('curvatureM', data=np.concatenate(curvatureM).astype(float))
        hf.create_dataset('curvatureG', data=np.concatenate(curvatureG).astype(float))
        hf.create_dataset('curvatureMAX', data=np.concatenate(curvatureMAX).astype(float))
        hf.create_dataset('curvatureMIN', data=np.concatenate(curvatureMIN).astype(float))
        hf.create_dataset('start', data=np.array(start).astype(int))
        hf.create_dataset('end', data=np.array(end).astype(int))
        hf.create_dataset('pid', data=np.array(pid).astype(int))






