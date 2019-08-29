import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData
from torch.utils import data
from config import n_trian

CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])


class Scannetloader(data.Dataset):
    def __init__(self, file_root,max_point=4096*2,n_class=20,is_train=True):
        self.file_root = file_root
        self.is_train = is_train
        self.data_list,self.label_list = self.get_data()
        self.max_point = max_point
        self.n_class = n_class

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        plydata = PlyData.read(self.data_list[index])
        plylabel = PlyData.read(self.label_list[index])

        pc = plydata['vertex'].data  # pc.shape = (n_point,)
        lbl = plylabel['vertex'].data

        label = lbl['label'].astype(np.int8) # label.shape = (n_point,) , label.max() >= 40,n_class=20
        # add 7-9dims
        zero_h = np.zeros_like(pc['x'])
        points = np.vstack((pc['x'], pc['y'],pc['z'],pc['red'],pc['green'],pc['blue'],pc['x'], pc['y'],pc['z'])).transpose()  # shape=(n_point,6), RGB未normalize

        label = self.convert_label(label)

        points,label = self.reduce_points(points,label)

        points = self.normalizer(points)

        # to numpy
        points = torch.from_numpy(points).float()
        label = torch.from_numpy(label).float()

        return points, label

    def normalizer(self,points):
        max_xyz = np.max(points[:,:3],axis=0)
        points[:,:3] = points[:,:3] / max_xyz
        points[:,3:6] = points[:,3:6] / 255.
        points[:, 6:] = points[:, 6:] / 200

        return points


    def reduce_points(self,points,label):
        '''
        reduce nums of point to max_point
        :param points:
        :param label:
        :return: points, label
        '''
        choice = np.random.choice(points.shape[0],size=self.max_point,replace=False)  # random choice max_point element
        points = points[choice,:]
        label = label[choice]

        return points,label

    def convert_label(self,labels):
        '''
        convert vaild label(1->40) to (1 -> 20)
        :param labels:
        :return: labels
        '''
        CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                        'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                        'bathtub', 'otherfurniture']
        VALID_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        _labels = np.zeros_like(labels)
        for i in range(self.n_class):
            _labels[labels==VALID_IDS[i]] = i

        labels = _labels

        return labels

    def get_data(self):
        # is train
        assert os.path.isdir(os.path.join(self.file_root, "scans")), "not found scans"
        #assert os.path.isdir(os.path.join(self.file_root, "scans_test")), "not found scans_test"

        data_list = []
        label_list = []
        file_list = os.walk(os.path.join(self.file_root, "scans"))
        for dirpath, _, filenames in file_list:
            for f_name in filenames:
                if f_name[-18:] == "clean_2.labels.ply":
                    label_list.append(os.path.join(dirpath,f_name))
                elif f_name[-11:] == "clean_2.ply":
                    data_list.append(os.path.join(dirpath,f_name))

        n_split = int(len(data_list)/n_trian)
        if self.is_train:
            data_list = data_list[n_split:]  # train set (2/3)
            label_list = label_list[n_split:]
        else:
            data_list = data_list[:n_split]  # test set (1/3)
            label_list = label_list[:n_split]

        return data_list,label_list


def visible(points, labels):
    """
    :param points: # shape = (4096,9)  data_batch[2,:,:]
    :param labels:   label_batch[2,:]
    :return:
    """

    skip = 1  # Skip every n points

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    point_range = range(0, points.shape[0], skip)  # skip points to prevent crash
    ax.scatter(points[point_range, 0],  # x
               points[point_range, 1],  # y
               points[point_range, 2],  # z
               c=labels[point_range],  # height data for color
               cmap='Spectral',
               alpha=1,
               s=2,
               marker=".")
    ax.axis('scaled')  # {equal, scaled}

    # show color bar
    plt.figure(2)
    range_cmap = [[i for i in range(len(CLASS_LABELS))]]
    plt.imshow(range_cmap, cmap='Spectral')
    plt.xticks(range_cmap[0], CLASS_LABELS, rotation=50)
    plt.show()

if __name__ == '__main__':
    origin_root = r'C:\Users\PC\Desktop\scannet'  # root of ScanNet dir
    loader = Scannetloader(origin_root,is_train=True,max_point=4096*3)
    trainloader = data.DataLoader(loader,batch_size=1,shuffle=True,num_workers=1)
    print(len(trainloader))
    for i,batch in enumerate(trainloader):
        points = np.array(batch[0])
        labels = np.array(batch[1])
        print(labels.min())
        visible(points[0,...],labels[0,...])

