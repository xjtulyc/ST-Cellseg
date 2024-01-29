import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

import stumap as umap

Xdata = np.load('stumap/Xdata.npy')
with open('stumap/ngc.pkl', 'rb') as f:
    ngc = pickle.load(f)
    pass
max_idx = 50261
idx_range = np.array([[idx] for idx in range(max_idx)])
reducer = umap.UMAP(metric='clustermap')
embedding = reducer.fit_transform(idx_range)
print(embedding.shape)
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
)
plt.show()
# DPC+K-means
kmeans = KMeans(n_clusters=10, random_state=10)
kmeans.fit(embedding)
y_predict = kmeans.predict(embedding)
plt.scatter(embedding[:, 0],
            embedding[:, 1],
            c=y_predict)
plt.show()

dbscan = DBSCAN(eps=0.045, min_samples=1)
dbscan.fit(embedding)
y_predict = dbscan.fit_predict(embedding)
plt.scatter(embedding[:, 0],
            embedding[:, 1],
            c=y_predict)
plt.show()

plt.scatter(Xdata[:, 0],
            Xdata[:, 1],
            c=y_predict,
            s=0.1)
plt.show()


def reject_outliers(data, m=4):
    test = abs(data - np.mean(data, axis=0)) < m * np.std(data, axis=0)
    list = [i[0] and i[1] for i in test]

    return data[list, :]

# cell_ids = y_predict
# cells_unique = np.unique(cell_ids)
# spots_repr = np.array(Xdata[['spot_location_2', 'spot_location_1']])
# cells_unique = cells_unique[cells_unique >= 0]
# img_res = np.zeros(dapi_stacked.shape)
# spots_repr[:, [0, 1]] = spots_repr[:, [1, 0]]
# Nlabels = cells_unique.shape[0]
# hulls = []
# coords = []
# num_cells = 0
# print('Creat cell convex hull')
# for i in cells_unique:
#     curr_coords = spots_repr[cell_ids == i, 0:2]
#     curr_coords = reject_outliers(curr_coords)
#     if curr_coords.shape[0] < 100000 and curr_coords.shape[0] > 50:
#         num_cells += 1
#         hulls.append(ConvexHull(curr_coords))
#         coords.append(curr_coords)
# print("Used %d / %d" % (num_cells, Nlabels))
#
# if self.num_dims == 2:
#     dapi_2D = self.dapi
# else:
#     dapi_2D = np.sum(self.dapi, axis=2)
#
# plt.figure(figsize=(figscale * width / float(height), figscale))
# polys = [hull_to_polygon(h, k) for h in hulls]
# if good_cells is not None:
#     polys = [p for i, p in enumerate(polys) if i in good_cells]
# p = PatchCollection(polys, alpha=alpha, cmap='tab20', edgecolor='k', linewidth=0.5)
# colors = cell_ids
# if vmin or vmax is not None:
#     p.set_array(colors)
#     p.set_clim(vmin=vmin, vmax=vmax)
# else:
#     if rescale_colors:
#         p.set_array(colors + 1)
#         p.set_clim(vmin=0, vmax=max(colors + 1))
#     else:
#
#         p.set_array(colors)
#         p.set_clim(vmin=0, vmax=max(colors))
#         # dapi_2D = (dapi_2D > 0).astype(np.int)
# plt.imshow(dapi_2D, cmap=plt.cm.gray_r, alpha=0.35, origin='lower')
# plt.gca().add_collection(p)
# # plot decision graph to set params `density_threshold`, `distance_threshold`.
# dpca = DensityPeakCluster(density_threshold=8, distance_threshold=5, anormal=False)
#
# # fit model
# dpca.fit(np.array(embedding, dtype=np.float16))
#
# # print predict label
# print(dpca.labels_)
#
# # plot cluster
# dpca.plot("all", title='embedding', save_path="")
