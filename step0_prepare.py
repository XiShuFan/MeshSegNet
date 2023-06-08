"""
构造MeshSegNet数据集
"""

import os
import vtk
from shutil import copyfile

def convert(dataset: str, img_path: str, label_path: str):
    """
    提供原始数据路径和目标数据路径
    """
    reader = vtk.vtkPLYReader()
    writer = vtk.vtkPolyDataWriter()
    # 当前文件夹下的所有子文件夹
    folders = [name for name in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, name))]

    for folder in folders:
        files = [name for name in os.listdir(os.path.join(dataset, folder))]

        for file in files:
            reader.SetFileName(os.path.join(dataset, folder, file))
            reader.Update()

            # 将PLY数据转换为VTK数据
            ply_data = reader.GetOutput()

            if 'label' in file:
                writer.SetFileName(os.path.join(label_path, folder.split('_')[0] + '_' + file.split('.')[0] + '.vtk'))

            else:
                writer.SetFileName(os.path.join(img_path, folder.split('_')[0] + '_' + file.split('.')[0] + '.vtk'))
            writer.SetInputData(ply_data)
            writer.Write()



def copyto(dataset: str, img_path: str, label_path: str):
    """
    拷贝原始文件到其他文件夹
    """
    # 当前文件夹下的所有子文件夹
    folders = [name for name in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, name))]

    for folder in folders:
        files = [name for name in os.listdir(os.path.join(dataset, folder))]

        for file in files:
            if 'label' in file:
                copyfile(os.path.join(dataset, folder, file),
                         os.path.join(label_path, folder.split('_')[0] + '_' + file.split('.')[0] + '.ply'))
            else:
                copyfile(os.path.join(dataset, folder, file),
                         os.path.join(img_path, folder.split('_')[0] + '_' + file.split('.')[0] + '.ply'))


def main():
    datasets = ['/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/OralScan/dataset_labelled_cell_color_downsampled_10000_colorrefine']
    img_path = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/OralScan/img'
    label_path = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/OralScan/label'

    # datasets = ['D:\\users\\xsf\\Dataset\\OralScan\\dataset_labelled_cell_color_downsampled_50000_colorrefine']
    # img_path = 'D:\\users\\xsf\\Dataset\\OralScan\\img'
    # label_path = 'D:\\users\\xsf\\Dataset\\OralScan\\label'

    for dataset in datasets:
        copyto(dataset, img_path, label_path)


if __name__ == '__main__':
    main()