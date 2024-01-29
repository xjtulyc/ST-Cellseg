import os.path

import numpy as np
import pandas as pd
import scipy.io
import tifffile


def dataloader_STARmap_human_cardiac_organoid(dataset_path=r'../dataset/STARmap_human_cardiac_organoid'):
    # read spots
    # read from *.mat
    mat = scipy.io.loadmat(os.path.join(dataset_path, 'allPoints.mat'))
    # read dapi
    dapi = tifffile.imread(os.path.join(dataset_path, 'round1_dapi.tiff'))
    dapi = np.transpose(dapi, (1, 2, 0))

    # print(mat)
    # read gene id in mat['allReads']
    gene = mat['allReads'].astype('int')
    gene = gene - np.min(gene) + 1

    # get gene annotation for barcode in mat['allReads']
    # LYC: 8 types
    gene_list = pd.DataFrame(['TNNI1', 'MYH7', 'MYL7', 'ATP2A2', 'NANOG', 'EOMES', 'CS44', 'TBXT'])

    # read spots in mat['allPoints']
    spots = pd.DataFrame(mat['allPoints'], columns=['spot_location_1', 'spot_location_2', 'spot_location_3'])
    spots['gene'] = gene
    return spots, dapi, gene_list


def dataloader_STARmap_MousePlacenta(dataset_path=r'../dataset/STARmap_MousePlacenta'):
    # read spots
    data = pd.read_csv(os.path.join(dataset_path, 'Spot_meta.csv'))
    Spot_gene_barcode, Spot_location_x, Spot_location_y, Spot_cell_id = data['Spot_gene_barcode'], \
                                                                        data['Spot_location_x'], \
                                                                        data['Spot_location_y'], \
                                                                        data['Spot_cell_id']
    gene_annotation_for_cellexpr = pd.read_csv(os.path.join(dataset_path, 'gene_annotation_for_cellexpr.csv'))
    spots = pd.concat([Spot_location_x, Spot_location_y, Spot_gene_barcode], axis=1)
    spots.rename(columns={'Spot_gene_barcode': 'gene',
                          'Spot_location_x': 'spot_location_2',
                          'Spot_location_y': 'spot_location_1'}, inplace=True)
    label = Spot_cell_id
    gene_list = gene_annotation_for_cellexpr
    # read dapi
    dapi = tifffile.imread(os.path.join(dataset_path, 'dapi_trimmed.tif'))
    dapi = np.transpose(dapi, (1, 0))
    return spots, dapi, gene_list, label


if __name__ == '__main__':
    # spots_car, dapi_car, gene_list_car = dataloader_STARmap_human_cardiac_organoid()
    spots, dapi, gene_list, label = dataloader_STARmap_MousePlacenta()
    pass
