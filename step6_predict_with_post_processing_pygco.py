import os
import random

import numpy as np
import torch
import torch.nn as nn
from meshsegnet_bn import *
import vedo
import pandas as pd
from losses_and_metrics_for_mesh import *
from scipy.spatial import distance_matrix
import scipy.io as sio
import shutil
import time
from sklearn.svm import SVC # uncomment this line if you don't install thudersvm
# from thundersvm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from pygco import cut_from_graph
import utils


def assign_color(mesh):
    """
    按照mesh的标签，给点上色
    """
    labels = mesh.celldata['Label']
    cells = mesh.cells()

    # 可视化结果
    visualize_mesh = mesh.clone()

    for idx, label in enumerate(labels):
        cell = cells[idx]
        # 设置点的颜色
        if label == 1:
            color = np.array([255, 255, 0, 255])
        else:
            color = np.array([255, 255, 255, 255])
        visualize_mesh.pointcolors[cell[0]] = color
        visualize_mesh.pointcolors[cell[1]] = color
        visualize_mesh.pointcolors[cell[2]] = color
    return visualize_mesh


if __name__ == '__main__':

    gpu_id = utils.get_avail_gpu()
    torch.cuda.set_device(gpu_id) # assign which gpu will be used (only linux works)

    # 上采样方法选择K最近邻
    # upsampling_method = 'SVM'
    upsampling_method = 'KNN'

    model_path = './models/coarse_1'
    model_name = 'MeshSegNet_1_classes_best.tar'

    # 需要修改这里的文件
    mesh_path = '/media/why/新加卷/xsf/Dataset/ply_file_cell_color_manifold'
    sample_filenames = os.listdir(mesh_path)
    # 测试一部分就行
    sample_filenames = random.sample(sample_filenames, 100)

    output_path = '/media/why/新加卷/xsf/Dataset/coarse_seg_result'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # 只有两类，牙齿和非牙齿
    num_classes = 2
    num_channels = 18

    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)

    # load trained model
    checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model = model.to(device, dtype=torch.float)

    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


    # Predicting
    model.eval()
    with torch.no_grad():
        for i_sample in sample_filenames:

            start_time = time.time()

            print('Predicting Sample filename: {}'.format(i_sample))
            # read image and label (annotation)
            mesh = vedo.load(os.path.join(mesh_path, i_sample))

            # pre-processing: downsampling
            print('\tDownsampling...')
            target_num = 10000
            ratio = target_num/mesh.ncells # calculate ratio
            mesh_d = mesh.clone()
            # TODO：可以调用vedo的下采样方法
            mesh_d.decimate(fraction=ratio)
            predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)

            # move mesh to origin
            print('\tPredicting...')
            points = mesh_d.points()
            mean_cell_centers = mesh_d.center_of_mass()
            points[:, 0:3] -= mean_cell_centers[0:3]

            ids = np.array(mesh_d.faces())
            cells = points[ids].reshape(mesh_d.ncells, 9).astype(dtype='float32')

            # customized normal calculation; the vtk/vedo build-in function will change number of points
            mesh_d.compute_normals()
            normals = mesh_d.celldata['Normals']

            # move mesh to origin
            barycenters = mesh_d.cell_centers() # don't need to copy
            barycenters -= mean_cell_centers[0:3]

            #normalized data
            maxs = points.max(axis=0)
            mins = points.min(axis=0)
            means = points.mean(axis=0)
            stds = points.std(axis=0)
            nmeans = normals.mean(axis=0)
            nstds = normals.std(axis=0)

            for i in range(3):
                cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
                cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
                cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
                barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
                normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

            X = np.column_stack((cells, barycenters, normals))

            # computing A_S and A_L
            A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            D = distance_matrix(X[:, 9:12], X[:, 9:12])
            A_S[D<0.1] = 1.0
            A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            A_L[D<0.2] = 1.0
            A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            # 添加面片密度信息
            M1 = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            M2 = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            M3 = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            M1[D < 0.05] = 1
            M2[D < 0.1] = 1
            M3[D < 0.2] = 1

            # 计算三个尺度下面片的密度，并且归一
            m1 = M1.sum(axis=1) / (np.sum(M1) + 1e-5)
            m1 = (m1 - min(m1)) / (max(m1) - min(m1) + 1e-5)
            m2 = M2.sum(axis=1) / (np.sum(M2) + 1e-5)
            m2 = (m2 - min(m2)) / (max(m2) - min(m2) + 1e-5)
            m3 = M3.sum(axis=1) / (np.sum(M3) + 1e-5)
            m3 = (m3 - min(m3)) / (max(m3) - min(m3) + 1e-5)

            # 把密度信息添加到输入
            X = np.column_stack((X, m1, m2, m3))

            # numpy -> torch.tensor
            X = X.transpose(1, 0)
            X = X.reshape([1, X.shape[0], X.shape[1]])
            X = torch.from_numpy(X).to(device, dtype=torch.float)
            A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
            A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
            A_S = torch.from_numpy(A_S).to(device, dtype=torch.float)
            A_L = torch.from_numpy(A_L).to(device, dtype=torch.float)

            tensor_prob_output = model(X, A_S, A_L).to(device, dtype=torch.float)
            patch_prob_output = tensor_prob_output.cpu().numpy()

            for i_label in range(num_classes):
                predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

            # output downsampled predicted labels
            mesh2 = mesh_d.clone()
            mesh2.celldata['Label'] = predicted_labels_d
            mesh2.celldata.select('Label')
            mesh2 = assign_color(mesh2)
            vedo.write(mesh2, os.path.join(output_path, '{}_d_predicted.ply'.format(i_sample[:-4])), binary=False) # 这里要么记录标签，要么给面片附上颜色

            # refinement
            print('\tRefining by pygco...')
            round_factor = 100
            patch_prob_output[patch_prob_output<1.0e-6] = 1.0e-6

            # unaries
            unaries = -round_factor * np.log10(patch_prob_output)
            unaries = unaries.astype(np.int32)
            unaries = unaries.reshape(-1, num_classes)

            # parawise
            pairwise = (1 - np.eye(num_classes, dtype=np.int32))

            #edges
            normals = mesh_d.celldata['Normals'].copy() # need to copy, they use the same memory address
            barycenters = mesh_d.cell_centers() # don't need to copy
            cell_ids = np.asarray(mesh_d.faces())

            lambda_c = 30
            edges = np.empty([1, 3], order='C')
            min_theta = 360
            max_theta = 0
            for i_node in range(cells.shape[0]):
                # Find neighbors
                nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
                nei_id = np.where(nei==2)
                for i_nei in nei_id[0][:]:
                    if i_node < i_nei: # 计算法向量的夹角余弦值，这里判断小于是为了避免重复添加邻居对
                        cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
                        if cos_theta >= 1.0:
                            cos_theta = 0.9999
                        theta = np.arccos(cos_theta) # 得到夹角

                        # 我感觉不能用theta > np.pi/2.0 来判断是否是凹面，凸面也可以满足这个条件判断
                        min_theta = min(min_theta, theta / np.pi * 180)
                        max_theta = max(max_theta, theta / np.pi * 180)
                        phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :]) # 这里计算的是邻居面片的欧式距离

                        # 如果夹角很大，边的权重反而小；如果夹角很小，边的权重反而很大
                        if theta > np.pi/2.0:
                            edges = np.concatenate((edges, np.array([i_node, i_nei, -np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
                        else:
                            beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                            edges = np.concatenate((edges, np.array([i_node, i_nei, -beta*np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
            edges = np.delete(edges, 0, 0) # 删除第一个没用的占位
            edges[:, 2] *= lambda_c*round_factor
            edges = edges.astype(np.int32)

            print(f'min_theta {min_theta} max_theta {max_theta}')

            # 最核心的这一步无法debug
            refine_labels = cut_from_graph(edges, unaries, pairwise)
            refine_labels = refine_labels.reshape([-1, 1])

            # output refined result
            mesh3 = mesh_d.clone()
            mesh3.celldata['Label'] = refine_labels
            mesh3.celldata.select('Label')
            mesh3 = assign_color(mesh3)
            vedo.write(mesh3, os.path.join(output_path, '{}_d_predicted_refined.ply'.format(i_sample[:-4])), binary=False)

            # upsampling
            print('\tUpsampling...')
            # TODO: 这里要保证原始精度，就别下采样了
            # if mesh.ncells > 50000: target_num = 50000 # set max number of cells
            # ratio = target_num/mesh.ncells # calculate ratio
            # mesh.decimate(fraction=ratio)
            # print('Original contains too many cells, simpify to {} cells'.format(mesh.ncells))

            # get fine_cells
            barycenters = mesh3.cell_centers() # don't need to copy
            fine_barycenters = mesh.cell_centers() # don't need to copy


            # 直接上采样面片标签啊，我没想到这种方法

            if upsampling_method == 'SVM':
                clf = SVC(kernel='rbf', gamma='auto')
                # train SVM
                clf.fit(barycenters, np.ravel(refine_labels))
                fine_labels = clf.predict(fine_barycenters)
                fine_labels = fine_labels.reshape(-1, 1)
            elif upsampling_method == 'KNN':
                neigh = KNeighborsClassifier(n_neighbors=3)
                # train KNN
                neigh.fit(barycenters, np.ravel(refine_labels))
                fine_labels = neigh.predict(fine_barycenters)
                fine_labels = fine_labels.reshape(-1, 1)

            mesh.celldata['Label'] = fine_labels
            mesh.celldata.select('Label')
            mesh = assign_color(mesh)
            vedo.write(mesh, os.path.join(output_path, '{}_predicted_refined.ply'.format(i_sample[:-4])), binary=False)

            end_time = time.time()
            print('Sample filename: {} completed'.format(i_sample))
            print('\tcomputing time: {0:.2f} sec'.format(end_time-start_time))


# 上采样算法解释
"""
其中，SVM（Support Vector Machine，支持向量机）是一种机器学习算法，它可以用于分类和回归问题。
这里的SVM用于分类问题，使用径向基函数（RBF）作为核函数，并通过训练来构建一个分类模型。
具体来说，通过将原始数据点映射到高维空间中，SVM可以找到最佳的超平面来区分数据点。
在这段代码中，SVM算法将barycenters作为输入数据，refine_labels作为对应的标签，通过训练来构建一个分类模型。
然后使用该模型来对fine_barycenters进行预测，得到相应的标签，存储在fine_labels中。

另外一种算法是KNN（K-Nearest Neighbor，K近邻），也是一种常见的分类算法。
在KNN算法中，对于新的数据点，将其与训练集中的所有数据点进行比较，选择距离最近的K个数据点，根据这K个数据点的类别进行投票，最终确定新数据点的类别。
这里的KNN算法使用sklearn库中的KNeighborsClassifier实现，其中n_neighbors参数设置为3，意味着选择距离最近的3个数据点进行投票。
同样地，KNN算法将barycenters作为输入数据，refine_labels作为对应的标签，通过训练来构建一个分类模型。
然后使用该模型来对fine_barycenters进行预测，得到相应的标签，存储在fine_labels中。
"""