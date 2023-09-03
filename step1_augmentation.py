import os
import numpy as np
import vtk
from vedo import *
from vedo import Mesh


def GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                               translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                               scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]):
    '''
    为了得到足够多的训练数据，对vtp数据进行随机旋转、平移、缩放
    get transformation matrix (4*4)
    return: vtkMatrix4x4
    '''
    Trans = vtk.vtkTransform()

    ry_flag = np.random.randint(0, 2)  # if 0, no rotate
    rx_flag = np.random.randint(0, 2)  # if 0, no rotate
    rz_flag = np.random.randint(0, 2)  # if 0, no rotate
    if ry_flag == 1:
        # rotate along Yth axis
        Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))
    if rx_flag == 1:
        # rotate along Xth axis
        Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))
    if rz_flag == 1:
        # rotate along Zth axis
        Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))

    trans_flag = np.random.randint(0, 2)  # if 0, no translate
    if trans_flag == 1:
        Trans.Translate([np.random.uniform(translate_X[0], translate_X[1]),
                         np.random.uniform(translate_Y[0], translate_Y[1]),
                         np.random.uniform(translate_Z[0], translate_Z[1])])

    scale_flag = np.random.randint(0, 2)
    if scale_flag == 1:
        Trans.Scale([np.random.uniform(scale_X[0], scale_X[1]),
                     np.random.uniform(scale_Y[0], scale_Y[1]),
                     np.random.uniform(scale_Z[0], scale_Z[1])])

    matrix = Trans.GetMatrix()

    return matrix


if __name__ == "__main__":
    # vtk数据路径
    vtk_path = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/OralScan_coarse_10000/train'
    # 输出数据增强之后的路径
    output_save_path = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/OralScan_coarse_10000/train_augmented'
    if not os.path.exists(output_save_path):
        os.mkdir(output_save_path)

    sample_list = os.listdir(vtk_path)

    # 对每一个vtk数据，做20次随机刚性变换
    num_augmentations = 10

    for sample in sample_list:
        print(sample)
        for aug in range(num_augmentations):
            output_file_name = sample.split('.')[0] + f'_{aug}.ply'
            vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                    translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                    scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2])
            mesh = load(os.path.join(vtk_path, sample))

            mesh.apply_transform(vtk_matrix)
            io.write(mesh, os.path.join(output_save_path, output_file_name), binary=False)

        # flipped mesh
        # for i_aug in range(num_augmentations):
        #     file_name = 'Sample_0{0}_d.vtp'.format(i_sample + 1000)
        #     output_file_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample + 1000)
        #     vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
        #                                             translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
        #                                             scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2])
        #     mesh = load(os.path.join(vtk_path, file_name))
        #     mesh.applyTransform(vtk_matrix)
        #     io.write(mesh, os.path.join(output_save_path, output_file_name), binary=False)
