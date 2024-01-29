from itertools import product

import numpy as np
import torch.nn as nn
from fastdist import fastdist
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from skimage.filters import threshold_otsu
from skimage.morphology import square, erosion, reconstruction
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def binarize_dapi(dapi, fast_preprocess, gauss_blur, sigma):
    '''
    Binarize raw dapi image

    params : - dapi (ndarray) = raw DAPI image

    returns : - dapi_binary (ndarray) = binarization of Dapi image
              - dapi_stacked (ndarray) =  2D stacked binarized image
    '''
    degree = len(dapi.shape)
    if gauss_blur:
        dapi = gaussian_filter(dapi, sigma=sigma)
    if fast_preprocess:
        if degree == 2:
            # binarize dapi
            thresh = threshold_otsu(dapi)
            binary = dapi >= thresh
            dapi_binary = np.array(binary).astype(float)
            dapi_stacked = dapi_binary
        else:
            dapi_binary = []
            for t in tqdm(np.arange(dapi.shape[2])):
                dapi_one_page = dapi[:, :, t]
                thresh = threshold_otsu(dapi_one_page)
                binary = dapi_one_page >= thresh
                dapi_binary.append(binary)  # z,y,x
                ### erosion on dapi binary
            dapi_binary = np.array(dapi_binary).transpose((1, 2, 0))  # y,x,z
            dapi_stacked = np.amax(dapi_binary, axis=2)

    else:
        if degree == 2:
            # binarize dapi
            dapi_marker = erosion(dapi, square(5))
            dapi_recon = reconstruction(dapi_marker, dapi)
            thresh = threshold_otsu(dapi_recon)
            binary = dapi_recon >= thresh
            dapi_binary = np.array(binary).astype(float)
            dapi_binary[dapi == 0] = False
            dapi_stacked = dapi_binary
        else:
            dapi_binary = []
            for t in tqdm(np.arange(dapi.shape[2])):
                dapi_one_page = dapi[:, :, t]
                dapi_marker = erosion(dapi_one_page, square(5))
                dapi_recon = reconstruction(dapi_marker, dapi_one_page)
                if len(np.unique(dapi_recon)) < 2:
                    thresh = 0
                    binary = dapi_recon >= thresh
                else:
                    thresh = threshold_otsu(dapi_recon)
                    binary = dapi_recon >= thresh
                dapi_binary.append(binary)  # z,y,x
                ### erosion on dapi binary
            dapi_binary = np.array(dapi_binary).transpose((1, 2, 0))  # y,x,z
            dapi_binary[dapi == 0] = False
            dapi_stacked = np.amax(dapi_binary, axis=2)

    return (dapi_binary, dapi_stacked)


def add_dapi_points(dapi_binary, dapi_grid_interval, spots_denoised, ngc, num_dims):
    '''
    Add sampled points for Binarized DAPI image to improve local densities

    params :    - dapi_binary (ndarray) = Binarized DAPI image
                - spots_denoised (dataframe) = denoised dataset
                - nodes (list of ints) = nodes of the StellarGraph
                - node_emb (ndarray) = node embeddings
    returns :   - spatial locations and ngc of all the points

    '''
    ### Sample dapi points
    sampling_mat = np.zeros(dapi_binary.shape)
    if num_dims == 3:
        for ii, jj, kk in product(range(sampling_mat.shape[0]), range(sampling_mat.shape[1]),
                                  range(sampling_mat.shape[2])):
            if ii % dapi_grid_interval == 0 and jj % dapi_grid_interval == 0 and kk % dapi_grid_interval == 0:
                sampling_mat[ii, jj, kk] = 1
        dapi_sampled = dapi_binary * sampling_mat
        dapi_coord = np.argwhere(dapi_sampled > 0)
        spots_points = spots_denoised.loc[:, ['spot_location_2', 'spot_location_1', 'spot_location_3']]
    else:
        for ii, jj in product(range(sampling_mat.shape[0]), range(sampling_mat.shape[1])):
            if ii % dapi_grid_interval == 0 and jj % dapi_grid_interval == 0:
                sampling_mat[ii, jj] = 1
        dapi_sampled = dapi_binary * sampling_mat
        dapi_coord = np.argwhere(dapi_sampled > 0)
        spots_points = spots_denoised.loc[:, ['spot_location_2', 'spot_location_1']]

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(spots_points)
    neigh_ind = knn.kneighbors(dapi_coord, 1, return_distance=False)

    ### Create dapi embedding thanks to the embedding of the nearest neighbor
    dapi_ngc = ngc[neigh_ind[:, 0]]

    ### Concatenate dapi embedding + <x,y,z> with spots embedding + <x,y,z>
    all_ngc = np.concatenate((ngc, dapi_ngc), axis=0)
    all_coord = np.concatenate((spots_points, dapi_coord), axis=0)
    return (all_coord, all_ngc)


def spearman_metric(x, y):
    '''
    Compute the spearman correlation as a metric
    '''
    return (spearmanr(x, y).correlation)


def DPC(dict_dataset, cell_num_threshold=0.01, use_genedis=True):
    '''
    Density Peak Clustering

    params :    - ngc (ndarray) = NGC vectors for each spot
                - spearman_metric (callable) = metric to use in the computation of genetic distance
    '''
    dapi = dict_dataset['dapi']
    dapi_binary, dapi_stacked = binarize_dapi(dapi, fast_preprocess=False, gauss_blur=False, sigma=1)
    # find nearest spots within radius and Compute spatial distance
    # add: consider z radius
    loss = nn.CrossEntropyLoss()
    # loss = torch.norm
    print('Compute spatial distance')
    xy_radius = 10
    z_radius = 7
    radius = max(xy_radius, z_radius)
    knn = NearestNeighbors(radius=radius)
    all_coord = np.array(dict_dataset['spots'].values[:, :3], dtype=np.float32)
    all_ngc = dict_dataset['ngc']
    dapi_grid_interval = 4
    all_coord, all_ngc = add_dapi_points(dapi_binary,
                                         dapi_grid_interval,
                                         dict_dataset['spots'],
                                         all_ngc,
                                         all_coord.shape[1])
    num_spots_with_dapi = all_coord.shape[0]
    print(f'After adding DAPI points, all spots:{num_spots_with_dapi}')
    print('DPC')
    knn.fit(all_coord)
    spatial_dist, spatial_nn_array = knn.radius_neighbors(all_coord, sort_results=True)
    # if self.num_dims == 3:
    #     if radius == self.xy_radius:
    #         smaller_radius = self.z_radius
    #     else:
    #         smaller_radius = self.xy_radius
    for indi, i in tqdm(enumerate(spatial_nn_array)):
        spatial_nn_array[indi] = i[all_coord[i, 2] - all_coord[indi, 2] <= radius]
        spatial_dist[indi] = spatial_dist[indi][all_coord[i, 2] - all_coord[indi, 2] <= radius]

    # Compute genetic distance with nearest spots within radius
    print('  Compute genetic distance')
    NGC_dist = spatial_dist.copy()
    # print(NGC_dist)
    # print(NGC_dist.shape)
    # NGC_dist = torch.Tensor(NGC_dist.astype(torch.float64))
    for i, j in tqdm(enumerate(spatial_nn_array)):
        NGC_dist[i] = fastdist.vector_to_matrix_distance(all_ngc[i, :], all_ngc[j, :], fastdist.euclidean, "euclidean")
        # NGC_dist[i] = loss(dict_dataset['kd_data'][i]['ngc'],
        #                    dict_dataset['kd_data'][j]['ngc']).numpy()
    # combine two dists
    if use_genedis:
        combine_dist = spatial_dist + NGC_dist / 10
    else:
        combine_dist = spatial_dist

    # compuete density rho and the nearest distance to a spot with higher density delta
    print('  Compute density rho and the nearest distance')
    rho = [np.exp(-np.square(i / (xy_radius * 0.4))).sum() for i in combine_dist]
    rho = np.array(rho)
    rho_descending_order = np.argsort(-rho)

    # to find delta and nneigh, compute knn within a large radius
    knn = NearestNeighbors(radius=radius * 5)
    knn.fit(all_coord)
    l_neigh_dist, l_neigh_array = knn.radius_neighbors(all_coord, sort_results=True)

    # find delta and nneigh for spots that exist within the large radius
    delta = np.zeros(rho_descending_order.shape)
    nneigh = np.zeros(rho_descending_order.shape)
    far_higher_rho = []
    for i, neigh_array_id in tqdm(enumerate(l_neigh_array)):
        try:
            loc = np.where(rho[neigh_array_id] > rho[neigh_array_id[0]])[0][0]
            delta[i] = l_neigh_dist[i][loc]
            nneigh[i] = neigh_array_id[loc]
        except IndexError:
            far_higher_rho.append(i)

    # find delta and nneigh for spots that don't exist within the large radius
    for i in tqdm(far_higher_rho):
        x_loc_i = np.where(rho_descending_order == i)[0][0]
        if x_loc_i > 0:
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(all_coord[rho_descending_order[:x_loc_i], :])
            dis, nearest_id = knn.kneighbors(all_coord[i, :].reshape(1, -1), return_distance=True)
            delta[i] = dis
            nneigh[i] = rho_descending_order[nearest_id]

    # assign delta for the largest density spot
    delta[rho_descending_order[0]] = np.max(delta)
    nneigh[rho_descending_order[0]] = -1

    # find cluster number
    number_cell = 0
    for numbertestid in range(2):
        if numbertestid == 0:
            lamda = rho * delta
        else:
            lamda = np.log(rho) * delta
        sort_lamda = -np.sort(-lamda)
        bin_index = range(0, len(dict_dataset['kd_data']), 10)
        start_value = sort_lamda[bin_index][:-1]
        middle_value = sort_lamda[bin_index][1:]
        change_value = start_value - middle_value
        curve = (change_value / (change_value[1] - change_value[-1]))

        for indi, i in enumerate(curve[:-1]):
            if i < cell_num_threshold and curve[indi + 1] < cell_num_threshold:
                number_cell = number_cell + (indi) * 10
                break
    number_cell = number_cell / 2
    if number_cell == 0:
        number_cell = 20
    print(f'  Find cell number:{number_cell}')

    # cellid[list12] = range(self.number_cell)  # range(cellid[list12].shape[0])
    # for i_value in tqdm(rho_descending_order):
    #     if cellid[int(i_value)] == -1:
    #         if cellid[int(nneigh[int(i_value)])] == -1:
    #             print('error')
    #         cellid[int(i_value)] = cellid[int(nneigh[int(i_value)])]

    # return (cellid)
    return number_cell

# embedding = np.load('embedding.npy')
# all_coord = np.load('stumap/Xdata.npy')
# print('  Compute spatial distance')
# xy_radius = 10
# z_radius = 7
# radius = max(xy_radius, z_radius)
#
# knn = NearestNeighbors(radius=radius)
# knn.fit(all_coord)
# spatial_dist, spatial_nn_array = knn.radius_neighbors(all_coord, sort_results=True)
# num_dims = 3
# if num_dims == 3:
#     if radius == xy_radius:
#         smaller_radius = z_radius
#     else:
#         smaller_radius = xy_radius
#     for indi, i in tqdm(enumerate(spatial_nn_array)):
#         spatial_nn_array[indi] = i[all_coord[i, 2] - all_coord[indi, 2] <= smaller_radius]
#         spatial_dist[indi] = spatial_dist[indi][all_coord[i, 2] - all_coord[indi, 2] <= smaller_radius]
