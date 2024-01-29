# ST-CellSeg: Cell segmentation for imaging-based spatial transcriptomics using multi-scale manifold learning

![](method_pipeline.jpg)

In this study, we proposed a spatial transcriptomic uniform manifold approximation (ST-Cellseg) algorithm. ST-Cellseg is an extension of uniform manifold approximation (UMAP) (McInnes et al., 2018) learning algorithm cell segmentation in spatial transcriptome. The proposed algorithm maps to the space for clustering of segmented cells by learning the manifold structure of the spatial transcriptome data. This algorithm is considered as a three-stage clustering algorithm. The first step is to learn the manifold structure of a fully connected graph which is constructed based on multi-scale distance metric of the spatial transcriptome. The second step is to find a low-dimensional spatial probability distribution representation that approximates the high-dimensional manifold structure. Finally, given the structure of manifold is learned in Euclidean space, cell segmentation is conducted based on the density clustering method (i.e., sample points are clustered in low-dimensional space).
run the code:

```
python main.py
```