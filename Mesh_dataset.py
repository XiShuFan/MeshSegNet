from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix
import time


class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, device, num_classes=15, patch_size=6000):
        """
        Args:
            h5_path (string): Path to the txt file with h5 files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = pd.read_csv(data_list_path, header=None)
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.device = device

        # TODO: 牙齿面片颜色对应的信息
        # 如果是粗糙分割，直接只有一个标签
        if num_classes == 2:
            self.color2label = {
                # 上颚颜色
                (170, 255, 127): ("aaff7f", "UL1", 1),
                (170, 255, 255): ("aaffff", "UL2", 1),
                (255, 255, 0): ("ffff00", "UL3", 1),
                (255, 170, 0): ("ffaa00", "UL4", 1),
                (170, 170, 255): ("aaaaff", "UL5", 1),
                (0, 170, 255): ("00aaff", "UL6", 1),
                (85, 170, 0): ("55aa00", "UL7", 1),
                (204, 204, 15): ("cccc0f", "UL8", 1),

                (255, 85, 255): ("ff55ff", "UR1", 1),
                (255, 85, 127): ("ff557f", "UR2", 1),
                (85, 170, 127): ("55aa7f", "UR3", 1),
                (255, 85, 0): ("ff5500", "UR4", 1),
                (0, 85, 255): ("0055ff", "UR5", 1),
                (170, 0, 0): ("aa0000", "UR6", 1),
                (73, 247, 235): ("49f7eb", "UR7", 1),
                (125, 18, 247): ("7d12f7", "UR8", 1),

                # 下颚颜色
                (240, 0, 0): ("f00000", "LL1", 1),
                (251, 255, 3): ("fbff03", "LL2", 1),
                (44, 251, 255): ("2cfbff", "LL3", 1),
                (241, 47, 255): ("f12fff", "LL4", 1),
                (125, 255, 155): ("7dff9b", "LL5", 1),
                (26, 125, 255): ("1a7dff", "LL6", 1),
                (255, 234, 157): ("ffea9d", "LL7", 1),
                (204, 126, 126): ("cc7e7e", "LL8", 1),

                (206, 129, 212): ("ce81d4", "LR1", 1),
                (45, 135, 66): ("2d8742", "LR2", 1),
                (185, 207, 45): ("b9cf2d", "LR3", 1),
                (69, 147, 207): ("4593cf", "LR4", 1),
                (207, 72, 104): ("cf4868", "LR5", 1),
                (4, 207, 4): ("04cf04", "LR6", 1),
                (35, 1, 207): ("2301cf", "LR7", 1),
                (82, 204, 169): ("52cca9", "LR8", 1),
            }
        else:
            self.color2label = {
                # 上颚颜色
                (170, 255, 127): ("aaff7f", "UL1", 1),
                (170, 255, 255): ("aaffff", "UL2", 2),
                (255, 255, 0): ("ffff00", "UL3", 3),
                (255, 170, 0): ("ffaa00", "UL4", 4),
                (170, 170, 255): ("aaaaff", "UL5", 5),
                (0, 170, 255): ("00aaff", "UL6", 6),
                (85, 170, 0): ("55aa00", "UL7", 7),
                (204, 204, 15): ("cccc0f", "UL8", 8),

                (255, 85, 255): ("ff55ff", "UR1", 9),
                (255, 85, 127): ("ff557f", "UR2", 10),
                (85, 170, 127): ("55aa7f", "UR3", 11),
                (255, 85, 0): ("ff5500", "UR4", 12),
                (0, 85, 255): ("0055ff", "UR5", 13),
                (170, 0, 0): ("aa0000", "UR6", 14),
                (73, 247, 235): ("49f7eb", "UR7", 15),
                (125, 18, 247): ("7d12f7", "UR8", 16),

                # 下颚颜色
                (240, 0, 0): ("f00000", "LL1", 1),
                (251, 255, 3): ("fbff03", "LL2", 2),
                (44, 251, 255): ("2cfbff", "LL3", 3),
                (241, 47, 255): ("f12fff", "LL4", 4),
                (125, 255, 155): ("7dff9b", "LL5", 5),
                (26, 125, 255): ("1a7dff", "LL6", 6),
                (255, 234, 157): ("ffea9d", "LL7", 7),
                (204, 126, 126): ("cc7e7e", "LL8", 8),

                (206, 129, 212): ("ce81d4", "LR1", 9),
                (45, 135, 66): ("2d8742", "LR2", 10),
                (185, 207, 45): ("b9cf2d", "LR3", 11),
                (69, 147, 207): ("4593cf", "LR4", 12),
                (207, 72, 104): ("cf4868", "LR5", 13),
                (4, 207, 4): ("04cf04", "LR6", 14),
                (35, 1, 207): ("2301cf", "LR7", 15),
                (82, 204, 169): ("52cca9", "LR8", 16),
            }

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i_mesh = self.data_list.iloc[idx][0]  # vtk file name

        # read vtk
        mesh = load(i_mesh)

        # 获取面片对应的标签
        labels = []
        for color in mesh.cellcolors:
            color = (color[0], color[1], color[2])
            if color in self.color2label:
                labels.append(self.color2label[color][2])
            else:
                labels.append(0)

        labels = np.array(labels).astype('int32')

        # 转换成一维向量
        labels = labels.reshape(-1, 1)

        # 将顶点坐标移动到坐标原点
        points = mesh.points()
        mean_cell_centers = mesh.center_of_mass()
        points[:, 0:3] -= mean_cell_centers[0:3]

        # 得到面片的三个顶点的坐标，转换成(面片数量,9)的shape
        ids = np.array(mesh.faces())
        cells = points[ids].reshape(mesh.ncells, 9).astype(dtype='float32')

        # 计算获得每个面的法向量
        # customized normal calculation; the vtk/vedo build-in function will change number of points
        mesh.compute_normals()
        normals = mesh.celldata['Normals']

        # 将面片的中心点坐标移动到原点
        # move mesh to origin
        barycenters = mesh.cell_centers()  # don't need to copy
        barycenters -= mean_cell_centers[0:3]

        # 计算所有点坐标的最大、最小、均值和方差 #计算面片法向量的均值和方差
        # normalized data
        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)

        # 对顶点坐标执行normalize操作，对面片的中心坐标执行归一化操作，对面片的法向量进行normalize操作
        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
            cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
            cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
            barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
            normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

        # 拼接顶点坐标（9）、面片中心点坐标（3）、面片法向量（3）
        X = np.column_stack((cells, barycenters, normals))
        Y = labels


        # calculate number of valid cells (tooth instead of gingiva)
        positive_idx = np.argwhere(labels > 0)[:, 0]  # tooth idx
        negative_idx = np.argwhere(labels == 0)[:, 0]  # gingiva idx
        all_idx = np.concatenate([positive_idx, negative_idx], axis=0)

        num_positive = len(positive_idx)  # number of selected tooth cells

        # 每个class随机选取相同个数的面片
        # selected_idx = np.ndarray(shape=(0,), dtype=int)
        # for i in range(1, self.num_classes):
        #     class_i_idx = np.argwhere(labels == i)[:, 0]
        #     class_i_selected_idx = np.random.choice(class_i_idx, size=min(self.patch_size // self.num_classes, len(class_i_idx)), replace=False)
        #     selected_idx = np.concatenate((selected_idx, class_i_selected_idx))
        # # 如果目前的面片个数不够，用0标签来填充
        # class_0_idx = np.argwhere(labels == 0)[:, 0]
        # class_0_selected_idx = np.random.choice(class_0_idx, size=self.patch_size - selected_idx.shape[0], replace=False)
        # selected_idx = np.concatenate((selected_idx, class_0_selected_idx))

        # 下面的采样方法不均衡
        # if num_positive > self.patch_size:  # all positive_idx in this patch
        #     positive_selected_idx = np.random.choice(positive_idx, size=self.patch_size, replace=False)
        #     selected_idx = positive_selected_idx
        # else:  # patch contains all positive_idx and some negative_idx
        #     num_negative = self.patch_size - num_positive  # number of selected gingiva cells
        #     positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
        #     negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
        #     selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

        # 随机选取
        selected_idx = np.random.choice(all_idx, size=self.patch_size, replace=False)

        selected_idx = np.sort(selected_idx, axis=None)

        # 初始化输入和输出，这里对面片进行了patch操作
        # initialize batch of input and label
        X_train = np.zeros([self.patch_size, X.shape[1]], dtype='float32')
        Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int32')
        S1 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        S2 = np.zeros([self.patch_size, self.patch_size], dtype='float32')

        X_train[:] = X[selected_idx, :]
        Y_train[:] = Y[selected_idx, :]

        if torch.cuda.is_available():
            TX = torch.as_tensor(X_train[:, 9:12], device=self.device)
            TD = torch.cdist(TX, TX)  # 计算两个tensor中所有向量之间的距离
            D = TD.cpu().numpy()
        else:
            D = distance_matrix(X_train[:, 9:12], X_train[:, 9:12])

        # TODO: 这里对float64数据类型计算邻接矩阵，消耗内存太大，[10000, 10000]的矩阵,32G都不够
        # 向量之间距离小于0.1的就设置成1
        S1[D < 0.1] = 1.0
        # S1 = S1 / (np.dot(np.sum(S1, axis=1, keepdims=True, dtype='float32'), np.ones((1, self.patch_size), dtype='float32')) + 1e-5)
        result = torch.matmul(torch.sum(torch.tensor(S1, device=self.device), dim=1, keepdim=True),
                              torch.ones((1, self.patch_size), device=self.device))
        S1 = torch.tensor(S1, device=self.device) / (result + 1e-5)
        S1 = S1.cpu().numpy()

        S2[D < 0.2] = 1.0
        # S2 = S2 / (np.dot(np.sum(S2, axis=1, keepdims=True, dtype='float32'), np.ones((1, self.patch_size), dtype='float32')) + 1e-5)
        result = torch.matmul(
            torch.sum(torch.tensor(S2, device=self.device), dim=1, keepdim=True),
            torch.ones((1, self.patch_size), device=self.device))
        S2 = torch.tensor(S2, device=self.device) / (result + 1e-5)
        S2 = S2.cpu().numpy()

        # 添加面片密度信息
        M1 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        M2 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        M3 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
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
        X_train = np.column_stack((X_train, m1, m2, m3))

        X_train = X_train.transpose(1, 0)
        Y_train = Y_train.transpose(1, 0)

        sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train),
                  'A_S': torch.from_numpy(S1), 'A_L': torch.from_numpy(S2)}
        return sample


if __name__ == '__main__':
    dataset = Mesh_Dataset('/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/OralScan/train_list_1.csv')
    print(dataset.__getitem__(0))
