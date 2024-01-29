import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
# from mayavi import mlab
from sklearn.cluster import DBSCAN

import stumap as umap


def convert(st_data):
    data = []
    for x in st_data:
        p_x = x['p']
        msngc_x = x['msngc'].flatten()
        d = np.array([p_x.shape[0]], dtype=np.int8)
        data.append(np.hstack((d, p_x, msngc_x)))
    return np.stack(data)


st_file = r'stumap/MSNGC/h_st.pkl'

with open(os.path.join(st_file), 'rb') as f:
    st_data = pickle.loads(f.read())
    pass

st_data = convert(st_data)
reducer = umap.UMAP(metric='euclidean',
                    n_neighbors=100)
embedding = reducer.fit_transform(st_data)
print(embedding.shape)

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    s=0.1
)
plt.show()

dbscan = DBSCAN(eps=0.009, min_samples=1)
dbscan.fit(embedding)
y_predict = dbscan.fit_predict(embedding)
plt.scatter(embedding[:, 0],
            embedding[:, 1],
            c=y_predict,
            s=0.1)
plt.show()

plt.scatter(st_data[:, 1],
            st_data[:, 2],
            c=y_predict,
            s=0.1)
plt.show()


# def viz_mayavi(points):
#     x = points[:, 0]  # x position of point
#     y = points[:, 1]  # y position of point
#     z = points[:, 2]  # z position of point
#     fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
#     mlab.points3d(x, y, z,
#                   z,
#                   mode="point",
#                   colormap='spectral',
#                   # color=(0, 1, 0),
#                   )
#     mlab.show()


# points = st_data[:, 1:4]
# viz_mayavi(points)
