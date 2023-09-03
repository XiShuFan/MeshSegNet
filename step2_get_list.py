import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == '__main__':

    data_path = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/OralScan_coarse_10000/train_augmented'
    output_path = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/OralScan_coarse_10000'

    # 是否进行了翻转增强
    with_flip = False

    sample_list = os.listdir(data_path)
    sample_list = [os.path.join(data_path, name) for name in sample_list]
    sample_list = np.asarray(sample_list)

    i_cv = 0
    # 交叉验证，K=5
    kf = KFold(n_splits=5, shuffle=False)
    for train_idx, val_idx in kf.split(sample_list):

        i_cv += 1
        print('Round:', i_cv)

        train_list, val_list = sample_list[train_idx], sample_list[val_idx]

        print('Training list:\n', train_list, '\nValidation list:\n', val_list)

        # training
        with open(os.path.join(output_path, 'train_list_{0}.csv'.format(i_cv)), 'w') as file:
            for f in train_list:
                file.write(f + '\n')

        # validation
        with open(os.path.join(output_path, 'val_list_{0}.csv'.format(i_cv)), 'w') as file:
            for f in val_list:
                file.write(f + '\n')

        print('--------------------------------------------')
        print('with flipped samples:', with_flip)
        print('# of train:', len(train_list))
        print('# of validation:', len(val_list))
        print('--------------------------------------------')
