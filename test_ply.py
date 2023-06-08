import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix


def read_ply_info(i_mesh: str):
    """
    读取ply文件中需要的数据
    """
    color2label = {
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




    # read vtk
    mesh = load(i_mesh)

    labels = []
    for color in mesh.cellcolors:
        color = (color[0], color[1], color[2])
        if color in color2label:
            labels.append(color2label[color][2])
        else:
            labels.append(0)

    labels = np.array(labels).astype('int32')
    labels = labels.reshape(-1, 1)

    # new way
    # move mesh to origin
    points = mesh.points()
    mean_cell_centers = mesh.center_of_mass()
    points[:, 0:3] -= mean_cell_centers[0:3]

    ids = np.array(mesh.faces())
    cells = points[ids].reshape(mesh.ncells, 9).astype(dtype='float32')

    # customized normal calculation; the vtk/vedo build-in function will change number of points
    mesh.compute_normals()
    normals = mesh.celldata['Normals']

    # move mesh to origin
    barycenters = mesh.cell_centers()  # don't need to copy
    barycenters -= mean_cell_centers[0:3]





if __name__ == "__main__":
    i_mesh = '/mnt/d/Dataset/OralScan/data7_upper_labeled_1.ply'
    read_ply_info(i_mesh)
