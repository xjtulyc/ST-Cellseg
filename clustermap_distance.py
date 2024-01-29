import os.path
from itertools import product

import numpy as np
import pandas as pd
import scipy.io
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import square, erosion, reconstruction
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from tqdm import tqdm
from tqdm import trange


# def plot_segmentation(figsize=(10, 10), plot_with_dapi=True, plot_dapi=False, method='clustermap', s=5,
#                       cmap=None, show=True, save=False, savepath=None):
#     cell_ids = spots[method]
#     cells_unique = np.unique(cell_ids)
#     try:
#         spots_repr = np.array(spots[['spot_location_2', 'spot_location_1']])[cell_ids >= 0]
#     except:
#         print('No cell is identified!')
#         return
#     cell_ids = cell_ids[cell_ids >= 0]
#
#     if method == 'clustermap':
#         if plot_with_dapi:
#             cell_ids = all_points_cellid
#             cells_unique = np.unique(cell_ids)
#             spots_repr = all_points[cell_ids >= 0]
#             cell_ids = cell_ids[cell_ids >= 0]
#     if len(cell_ids) == 0:
#         print('Error:cell id is empty!')
#         return
#     if not show:
#         plt.ioff()
#     plt.figure(figsize=figsize)
#     if cmap is None:
#         myList = []
#
#         cmap = np.random.rand(int(max(cell_ids) + 1), 3)
#
#     if plot_dapi:
#         if num_dims == 3:
#             plt.imshow(np.sum(dapi_binary, axis=2), origin='lower', cmap='binary_r')
#         elif num_dims == 2:
#             plt.imshow(dapi_binary, origin='lower', cmap='binary_r')
#         plt.scatter(spots_repr[:, 1], spots_repr[:, 0],
#                     c=cmap[[int(x) for x in cell_ids]], s=s)
#     else:
#         plt.scatter(spots_repr[:, 1], spots_repr[:, 0],
#                     c=cmap[[int(x) for x in cell_ids]], s=s)
#
#     plt.title(method)
#     if save:
#         plt.savefig(savepath)
#     if show:
#         plt.show()

def binarize_dapi(dapi, fast_preprocess, gauss_blur, sigma):
    """
    Binarize raw dapi image

    params : - dapi (ndarray) = raw DAPI image

    returns : - dapi_binary (ndarray) = binarization of Dapi image
              - dapi_stacked (ndarray) =  2D stacked binarized image
    """
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


def preprocessing_data(spots, dapi_grid_interval, dapi_binary, LOF, contamination, xy_radius, pct_filter):
    '''
    Apply preprocessing on spots, thanks to dapi.
    We remove the 10% spots with lowest density

    params :    - spots (dataframe) = spatial locations and gene identity
                - dapi_binary (ndarray) = binarized dapi image

    returns :   - spots (dataframe)
    '''

    sampling_mat = np.zeros(dapi_binary.shape)
    if len(dapi_binary.shape) == 3:
        for ii, jj, kk in product(range(sampling_mat.shape[0]), range(sampling_mat.shape[1]),
                                  range(sampling_mat.shape[2])):
            if ii % dapi_grid_interval == 1 and jj % dapi_grid_interval == 1 and kk % dapi_grid_interval == 1:
                sampling_mat[ii, jj, kk] = 1
        dapi_sampled = dapi_binary * sampling_mat
        dapi_coord = np.argwhere(dapi_sampled > 0)

        all_points = np.concatenate(
            (np.array(spots.loc[:, ['spot_location_2', 'spot_location_1', 'spot_location_3']]), dapi_coord), axis=0)

        # compute neighbors within radius for local density
        knn = NearestNeighbors(radius=xy_radius * 2)
        knn.fit(all_points)
        spots_array = np.array(spots.loc[:, ['spot_location_2', 'spot_location_1', 'spot_location_3']])
        neigh_dist, neigh_array = knn.radius_neighbors(spots_array)

        # global low-density removal
        dis_neighbors = [(ii * ii).sum(0) for ii in neigh_dist]
        thresh = np.percentile(dis_neighbors, pct_filter * 100)
        noisy_points = np.argwhere(dis_neighbors < thresh)[:, 0]
        spots['is_noise'] = 0
        spots.loc[noisy_points, 'is_noise'] = -1

        # LOF
        if LOF:
            res_num_neighbors = [i.shape[0] for i in neigh_array]
            thresh = np.percentile(res_num_neighbors, 10)
            clf = LocalOutlierFactor(n_neighbors=int(thresh), contamination=contamination)
            spots_array = np.array(spots.loc[:, ['spot_location_2', 'spot_location_1', 'spot_location_3']])
            y_pred = clf.fit_predict(spots_array)
            spots.loc[y_pred == -1, 'is_noise'] = -1

        # spots in DAPI as inliers
        inDAPI_points = [i[0] and i[1] and i[2] for i in zip(spots_array[:, 0] - 1 < dapi_binary.shape[0],
                                                             spots_array[:, 1] - 1 < dapi_binary.shape[1],
                                                             spots_array[:, 2] - 1 < dapi_binary.shape[2])]
        test = dapi_binary[
            (spots_array[:, 0] - 1)[inDAPI_points], (spots_array[:, 1] - 1)[inDAPI_points], (spots_array[:, 2] - 1)[
                inDAPI_points]]
        inx = 0
        for indi, i in enumerate(inDAPI_points):
            if i == True:
                inDAPI_points[indi] = test[inx]
                inx = inx + 1
        spots.loc[inDAPI_points, 'is_noise'] = 0
    else:
        for ii, jj in product(range(sampling_mat.shape[0]), range(sampling_mat.shape[1])):
            if ii % dapi_grid_interval == 1 and jj % dapi_grid_interval == 1:
                sampling_mat[ii, jj] = 1

        dapi_sampled = dapi_binary * sampling_mat
        dapi_coord = np.argwhere(dapi_sampled > 0)

        all_points = np.concatenate((np.array(spots.loc[:, ['spot_location_2', 'spot_location_1']]), dapi_coord),
                                    axis=0)

        # compute neighbors within radius for local density
        knn = NearestNeighbors(radius=xy_radius)
        knn.fit(all_points)
        spots_array = np.array(spots.loc[:, ['spot_location_2', 'spot_location_1']])
        neigh_dist, neigh_array = knn.radius_neighbors(spots_array)

        # global low-density removal
        dis_neighbors = [ii.sum(0) for ii in neigh_dist]
        res_num_neighbors = [ii.shape[0] for ii in neigh_array]

        thresh = np.percentile(dis_neighbors, pct_filter * 100)
        noisy_points = np.argwhere(dis_neighbors < thresh)[:, 0]
        spots['is_noise'] = 0
        spots.loc[noisy_points, 'is_noise'] = -1

        # LOF
        if LOF:
            thresh = np.percentile(res_num_neighbors, 10)
            clf = LocalOutlierFactor(n_neighbors=int(thresh), contamination=contamination)
            spots_array = np.array(spots.loc[:, ['spot_location_2', 'spot_location_1']])
            y_pred = clf.fit_predict(spots_array)
            spots.loc[y_pred == -1, 'is_noise'] = -1

        # spots in DAPI as inliers
        test = dapi_binary[list(spots_array[:, 0] - 1), list(spots_array[:, 1] - 1)]
        spots.loc[test == True, 'is_noise'] = 0

        inDAPI_points = [i[0] and i[1] for i in zip(spots_array[:, 0] - 1 < dapi_binary.shape[0],
                                                    spots_array[:, 1] - 1 < dapi_binary.shape[1])]
        test = dapi_binary[(spots_array[:, 0] - 1)[inDAPI_points], (spots_array[:, 1] - 1)[inDAPI_points]]
        inx = 0
        for indi, i in enumerate(inDAPI_points):
            if i == True:
                inDAPI_points[indi] = test[inx]
                inx = inx + 1
        spots.loc[inDAPI_points, 'is_noise'] = 0

    return (spots)


def preprocess(spots, dapi_binary, xy_radius, dapi_grid_interval=5, LOF=False, contamination=0.1, pct_filter=0.1):
    preprocessing_data(spots, dapi_grid_interval, dapi_binary, LOF, contamination, xy_radius,
                       pct_filter)
    pass


def get_distance_matrix(points):
    points_num = points.shape[0]
    print(points_num)
    distance_matrix = np.zeros((points_num, points_num), dtype=np.float16)
    for i in trange(points_num):
        for j in range(points_num):
            point_i = points[i]
            point_j = points[j]
            distance_matrix[i, j] = (np.sum((point_i - point_j) ** 2))
            pass
    return np.sqrt(distance_matrix)


def NGC(spots):
    '''
    Compute the NGC coordinates

    params :    - radius float) = radius for neighbors search
                - num_dim (int) = 2 or 3, number of dimensions used for cell segmentation
                - gene_list (1Darray) = list of genes used in the dataset

    returns :   NGC matrix. Each row is a NGC vector
    '''

    if num_dims == 3:
        radius = max(xy_radius, z_radius)
        X_data = np.array(spots[['spot_location_1', 'spot_location_2', 'spot_location_3']])
    else:
        radius = xy_radius
        X_data = np.array(spots[['spot_location_1', 'spot_location_2']])
    knn = NearestNeighbors(radius=radius)
    knn.fit(X_data)
    spot_number = spots.shape[0]
    res_dis, res_neighbors = knn.radius_neighbors(X_data, return_distance=True)
    if num_dims == 3:
        ### remove nearest spots outside z_radius
        if radius == xy_radius:
            smaller_radius = z_radius
        else:
            smaller_radius = xy_radius
        for indi, i in tqdm(enumerate(res_neighbors)):
            res_neighbors[indi] = i[X_data[i, 2] - X_data[indi, 2] <= smaller_radius]
            res_dis[indi] = res_dis[indi][X_data[i, 2] - X_data[indi, 2] <= smaller_radius]

    res_ngc = np.zeros((spot_number, len(gene_list)))
    for i in trange(spot_number):
        neighbors_i = res_neighbors[i]
        genes_neighbors_i = spots.loc[neighbors_i, :].groupby('gene').size()
        res_ngc[i, genes_neighbors_i.index.to_numpy() - np.min(gene_list)] = np.array(genes_neighbors_i)
        # res_ngc[i] /= len(neighbors_i)
    return res_ngc


def distance2(x, y):
    x_y = np.array(x - y, dtype=np.float32)
    return np.sqrt(np.sum(np.dot(x_y, x_y)))


def spearman_corr(x, y):
    norm_x = x - np.mean(x)
    norm_y = y - np.mean(y)
    corr = np.sum(np.array(np.dot(norm_x, norm_y), dtype=np.float32))
    s_x = np.sum(np.array(np.dot(norm_x, norm_x), dtype=np.float32))
    s_y = np.sum(np.array(np.dot(norm_y, norm_y), dtype=np.float32))
    corr = corr / np.sqrt(s_x * s_y)
    return corr


fast_preprocess = False
gauss_blur = False
sigma = 1
# read spots
dataset_path = r'dataset/STARmap_human_cardiac_organoid'
# read from *.mat
mat = scipy.io.loadmat(os.path.join(dataset_path, 'allPoints.mat'))
# read dapi
dapi = tifffile.imread(os.path.join(dataset_path, 'round1_dapi.tiff'))
dapi = np.transpose(dapi, (1, 2, 0))
dapi_binary, dapi_stacked = binarize_dapi(dapi, fast_preprocess, gauss_blur, sigma)
# print(mat)
# read gene id in mat['allReads']
gene = mat['allReads'].astype('int')
gene = gene - np.min(gene) + 1

# get gene annotation for barcode in mat['allReads']
# LYC: 8 types
genes = pd.DataFrame(['TNNI1', 'MYH7', 'MYL7', 'ATP2A2', 'NANOG', 'EOMES', 'CS44', 'TBXT'])

# read spots in mat['allPoints']
spots = pd.DataFrame(mat['allPoints'], columns=['spot_location_1', 'spot_location_2', 'spot_location_3'])
spots['gene'] = gene

data = np.array(spots)
Physical_coordinates = data[:, :3]
Gene_list = data[:, 3]
# distance_matrix = get_distance_matrix(Physical_coordinates)
# set radius parameters
# 设置超参数
# pixel
xy_radius = 10
z_radius = 7
num_gene = np.max(spots['gene'])
gene_list = np.arange(1, num_gene + 1)
num_dims = len(dapi.shape)
# find the noise points
pct_filter = 0.1
print('start preprocess')
preprocess(spots, dapi_binary, xy_radius, pct_filter=pct_filter)
print('end preprocess')
spots['is_noise'] = spots['is_noise'] + 1
# plot_segmentation(figsize=(4, 4), s=0.6, method='is_noise',
#                   cmap=np.array(((0, 1, 0), (1, 0, 0))),
#                   plot_dapi=True)
spots['is_noise'] = spots['is_noise'] - np.min(spots['is_noise']) - 1
min_spot_per_cell = 5
cell_num_threshold = 0.001
dapi_grid_interval = 4
add_dapi = True
use_genedis = True
spots_denoised = spots.loc[spots['is_noise'] == 0, :].copy()
if 'level_0' in spots.columns:
    spots_denoised = spots_denoised.drop('level_0', axis=1)
spots_denoised.reset_index(inplace=True)
print(f'After denoising, mRNA spots: {spots_denoised.shape[0]}')
ngc = NGC(spots_denoised)
print(f'NGC shape is ' + str(ngc.shape))


class ClusterMap_distance:
    def __init__(self):
        self.spots = spots_denoised
        self.ngc = ngc
        self.num_dims = num_dims
        self.xy_radius = xy_radius
        self.z_radius = z_radius
        if self.num_dims == 3:
            self.radius = max(self.xy_radius, self.z_radius)
            self.X_data = np.array(spots[['spot_location_1', 'spot_location_2', 'spot_location_3']])
        else:
            self.radius = self.xy_radius
            self.X_data = np.array(spots[['spot_location_1', 'spot_location_2']])

    def __len__(self):
        return self.ngc.shape[0]

    def get_distance(self, i, j):
        points_distance = distance2(self.X_data[i], self.X_data[j])
        ngc_distance = spearman_corr(self.ngc[i], self.ngc[j])
        cm_distance = None
        if points_distance == 0 and ngc_distance == 0:
            cm_distance = 0
        if points_distance != 0 and ngc_distance == 0:
            cm_distance = np.inf
        cm_distance = points_distance / ngc_distance
        return cm_distance


get_ClusterMap_distance = ClusterMap_distance()
