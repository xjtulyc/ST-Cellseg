import os
from collections import Counter

import numpy as np
import torch
from sklearn.neighbors import KDTree

from DPC import DPC
from stumap.MSNGC.data_preprocess.preprocess import dataloader_STARmap_human_cardiac_organoid


def build_kd_data(data):
    tree = KDTree(data)
    return tree


def save(data, path):
    torch.save(data, path)


def load(path):
    return torch.load(path)


def build_nearest_dataset(dataset_name, spots, dapi, gene, kdtree, k=50, save_data=False):
    dict_dataset = dict()
    dict_dataset['dataset_name'] = dataset_name
    dict_dataset['spots'] = spots
    dict_dataset['dapi'] = dapi
    dict_dataset['gene'] = gene
    dict_dataset['kdtree'] = kdtree
    dict_dataset['k'] = k
    kd_data = dict()
    for id, point in enumerate(np.array(spots.values[:, :3], dtype=np.float64)):
        dist, ind = tree.query(np.array([point]), k=k)
        kd_data[id] = {'dist': dist, 'ind': ind}
    dict_dataset['kd_data'] = kd_data
    dict_dataset = build_NGC(dict_dataset)
    if save_data:
        save_path = r'dataset/kd_data'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_file = os.path.join(save_path, dataset_name) + '.pth'
        save(dict_dataset, save_file)
        print(f"Save dict_dataset to {save_file}")
    # if not os.path.exists(save_path):
    #     save(dict_dataset, save_file)
    #     print(f"Save dict_dataset to {save_file}")
    return dict_dataset


def build_NGC(dict_dataset):
    spots = dict_dataset['spots']
    gene = np.array(spots.values[:, 3], dtype=np.float32)
    ngc_ = []
    for id in dict_dataset['kd_data']:
        ind = dict_dataset['kd_data'][id]['ind']
        ngc = gene[ind]
        ngc_count = Counter(ngc[0])
        # print(f"{id} ngc_count {ngc_count}")
        ngc_feature = np.zeros(dict_dataset['gene'].values.shape[0], dtype=np.float64)
        # print(ngc_count)
        for type in ngc_count:
            ngc_feature[int(type - 1)] = ngc_count[type]
        ngc_.append(ngc_feature)
        pass
    dict_dataset['ngc'] = np.array(ngc_)
    return dict_dataset


def k_neighbor(spots):
    pass


if __name__ == '__main__':
    dataset_ = r'dataset'
    dataset_name = r'STARmap_human_cardiac_organoid'
    dataset_path = os.path.join(dataset_, dataset_name)
    spots, dapi, gene = dataloader_STARmap_human_cardiac_organoid(dataset_path)
    data = np.array(spots.values[:, :3], dtype=np.float32)
    spots_number = data.shape[0]
    # data = data / data.max()
    tree = build_kd_data(data)
    # number = []
    cell_number = 1425
    dict_dataset = build_nearest_dataset(dataset_name, spots, dapi, gene, tree,
                                         k=int(spots_number / cell_number))
    cell_number = DPC(dict_dataset)
    # for i in range(5):
    #     dict_dataset = build_nearest_dataset(dataset_name, spots, dapi, gene, tree,
    #                                          k=int(spots_number / cell_number))
    #     cell_number = DPC(dict_dataset)
    #     number.append(cell_number)
    print(f"cell number {cell_number}")
    # dist, ind = tree.query(data[:1], k=3)  # k nearest neibor
    # # s = pickle.dumps(tree)
    # # tree_copy = pickle.loads(s)
    # tree.query_radius(data[:1], r=0.3, count_only=True)
    # ind = tree.query_radius(data[:1], r=0.3)
    # print(dict_dataset['kd_data'][0])
