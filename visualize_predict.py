"""
可视化预测结果
"""
import os

import vedo
import numpy as np




if __name__ == '__main__':
    predict_path = 'D:\\Dataset\\OralScan\\test_dataset\\predicts'
    visualize_path = 'D:\\Dataset\\OralScan\\test_dataset\\visualize'

    for file in os.listdir(predict_path):
        predict_mesh = vedo.load(os.path.join(predict_path, file))
        labels = predict_mesh.celldata['Label']

        points = predict_mesh.points()
        cells = predict_mesh.cells()

        # 可视化结果
        visualize_mesh = predict_mesh.clone()




        visualize_mesh.show()
        vedo.write(visualize_mesh, os.path.join(visualize_path, file), binary=False)







