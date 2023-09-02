import concurrent.futures
import os
import numpy as np
import torch
import torch.nn as nn
from meshsegnet_ln import *
import vedo
import pandas as pd
from losses_and_metrics_for_mesh import *
from scipy.spatial import distance_matrix
import utils
import time
from multiprocessing import cpu_count

label2color_upper = {
    # 上颚颜色
    1: ("aaff7f", "UL1", (170, 255, 127)),
    2: ("aaffff", "UL2", (170, 255, 255)),
    3: ("ffff00", "UL3", (255, 255, 0)),
    4: ("faa00", "UL4", (255, 170, 0)),
    5: ("aaaaff", "UL5", (170, 170, 255)),
    6: ("00aaff", "UL6", (0, 170, 255)),
    7: ("55aa00", "UL7", (85, 170, 0)),
    8: ("cccc0f", "UL8", (204, 204, 15)),
    9: ("ff55ff", "UR1", (255, 85, 255)),
    10: ("ff557f", "UR2", (255, 85, 127)),
    11: ("55aa7f", "UR3", (85, 170, 127)),
    12: ("ff5500", "UR4", (255, 85, 0)),
    13: ("0055ff", "UR5", (0, 85, 255)),
    14: ("aa0000", "UR6", (170, 0, 0)),
    15: ("49f7eb", "UR7", (73, 247, 235)),
    16: ("7d12f7", "UR8", (125, 18, 247))
}


label2color_lower = {
# 下颚颜色
    1: ("f00000", "LL1", (240, 0, 0)),
    2: ("fbff03", "LL2", (251, 255, 3)),
    3: ("2cfbff", "LL3", (44, 251, 255)),
    4: ("f12fff", "LL4", (241, 47, 255)),
    5: ("7dff9b", "LL5", (125, 255, 155)),
    6: ("1a7dff", "LL6", (26, 125, 255)),
    7: ("ffea9d", "LL7", (255, 234, 157)),
    8: ("cc7e7e", "LL8", (204, 126, 126)),
    9: ("ce81d4", "LR1", (206, 129, 212)),
    10: ("2d8742", "LR2", (45, 135, 66)),
    11: ("b9cf2d", "LR3", (185, 207, 45)),
    12: ("4593cf", "LR4", (69, 147, 207)),
    13: ("cf4868", "LR5", (207, 72, 104)),
    14: ("04cf04", "LR6", (4, 207, 4)),
    15: ("2301cf", "LR7", (35, 1, 207)),
    16: ("52cca9", "LR8", (82, 204, 169))
}


# 计算邻接矩阵效率太低了，放到GPU上计算
def distance_matrix_gpu(x, y, device):

    n, k = x.shape
    m, kk = y.shape

    assert k == kk

    result = torch.zeros((n, m), device=device)
    x = torch.tensor(x.astype('float32'), device=device)
    y = torch.tensor(y.astype('float32'), device=device)

    if n < m:
        for i in range(n):
            result[i, :] = torch.sqrt(torch.sum(torch.square(torch.abs(y - x[i])), dim=-1))
    else:
        for j in range(m):
            result[:, j] = torch.sqrt(torch.sum(torch.square(torch.abs(x - y[j])), dim=-1))

    return result.cpu().numpy()



def main():
    gpu_id = utils.get_avail_gpu()
    torch.cuda.set_device(gpu_id) # assign which gpu will be used (only linux works)

    model_path = './models/10'
    model_name = 'MeshSegNet_15_classes_best.tar'

    # 需要预测的mesh的路径
    # mesh_path = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/OralScan/val_dataset/meshes'
    mesh_path = 'D:\\users\\xsf\\Dataset\\OralScan\\test_dataset\\meshes'
    sample_filenames = os.listdir(mesh_path)
    # 输出路径
    # output_path = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/OralScan/val_dataset/predicts'
    output_path = 'D:\\users\\xsf\\Dataset\\OralScan\\test_dataset\\predicts'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    num_classes = 17
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
            mesh = vedo.load(os.path.join(mesh_path, i_sample))

            # 为了保证精度，只能下采样到50000
            if mesh.ncells > 30000:
                print('\tDownsampling...')
                target_num = 30000
                ratio = target_num/mesh.ncells # calculate ratio
                mesh_d = mesh.clone()
                mesh_d.decimate(fraction=ratio)
                predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)
            else:
                mesh_d = mesh.clone()
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

            """
            调用多线程预测口扫的每一部分，测试过行不通
            """
            # 创建一个线程池
            # with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            #     sub_len = X.shape[0] // cpu_count()
            #     # 提交多个任务，注意要均分这些面片
            #     tasks = [executor.submit(multi_inference, X[i::cpu_count(), ], model, device, i)
            #              for i in range(cpu_count())]
            #     # 等待所有任务完成
            #     concurrent.futures.wait(tasks)
            #     # 获取所有任务的结果
            #     patch_prob_output_sublist = [task.result() for task in tasks]
            # # 拼接结果，还原原来的面片位置
            # patch_prob_output = np.zeros((1, X.shape[0], num_classes), dtype='float32')
            # for i in range(cpu_count()):
            #     patch_prob_output[0, i::cpu_count(), ] = patch_prob_output_sublist[i]

            # computing A_S and A_L
            A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            distance_matrix_start = time.time()

            D = distance_matrix_gpu(X[:, 9:12], X[:, 9:12], device)

            distance_matrix_end = time.time()
            print(f'distance matrix time {(distance_matrix_end - distance_matrix_start) / 60} mins')

            A_S_L_start = time.time()
            # 放到GPU计算
            A_S[D < 0.1] = 1.0
            result = torch.matmul(torch.sum(torch.tensor(A_S, device=device), dim=1, keepdim=True),
                      torch.ones((1, X.shape[0]), device=device))
            A_S = torch.tensor(A_S, device=device) / (result + 1e-5)
            A_S = A_S.cpu().numpy()

            A_L[D < 0.2] = 1.0
            result = torch.matmul(
                torch.sum(torch.tensor(A_L, device=device), dim=1, keepdim=True),
                torch.ones((1, X.shape[0]), device=device))
            A_L = torch.tensor(A_L, device=device) / (result + 1e-5)
            A_L = A_L.cpu().numpy()

            A_S_L_end = time.time()
            print(f'A_S_L time cost {(A_S_L_end - A_S_L_start) / 60} mins')

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

            inference_start = time.time()
            tensor_prob_output = model(X, A_S, A_L).to(device, dtype=torch.float)
            inference_end = time.time()
            print(f'inference time {(inference_end - inference_start) / 60} mins')

            patch_prob_output = tensor_prob_output.detach().cpu().numpy()

            for i_label in range(num_classes):
                predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1) == i_label] = i_label

            # output downsampled predicted labels
            mesh2 = mesh_d.clone()
            mesh2.celldata['Label'] = predicted_labels_d
            cells = mesh2.cells()

            # 把相应的cell的三个点设置成对应的颜色
            for idx, label in enumerate(predicted_labels_d):
                label = label[0]
                if label in label2color_upper:
                    cell = cells[idx]
                    color = label2color_upper[label][-1]
                    color = np.array([color[0], color[1], color[2], 255])
                    # 设置点的颜色
                    mesh2.pointcolors[cell[0]] = color
                    mesh2.pointcolors[cell[1]] = color
                    mesh2.pointcolors[cell[2]] = color

            # mesh2.show()

            vedo.write(mesh2, os.path.join(output_path, '{}_d_predicted.ply'.format(i_sample[:-4])), binary=False)

            end_time = time.time()
            time_cost = end_time - start_time
            print('Sample filename: {0} completed, time cost {1} mins'.format(i_sample, time_cost / 60))


if __name__ == "__main__":
    main()
