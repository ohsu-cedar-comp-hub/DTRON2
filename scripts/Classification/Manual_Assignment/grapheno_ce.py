"""
AUTHOR: Christopher Z. Eddy (eddyc@ohsu.edu)
Notes: Full credit for this code to Erik Burlingame. I have slightly modified this script from Erik's Grapheno codebase, since there were previously incompatible libraries. 
"""
import os
import time
import cuml
import cudf
import cugraph
import dask_cudf
import cupy as cp
from dask.distributed import Client
from dask_cuda import LocalCUDACluster, initialize
#import cugraph.dask.comms.comms as Comms
from cuml.neighbors import NearestNeighbors as NN
from cuml.dask.neighbors import NearestNeighbors as DaskNN

import numpy as np
from tqdm.notebook import tqdm

def generate_dummy_data(n_samples = 5000000,
                        n_features = 20,
                        centers = 30,
                        cluster_std=3.0):
    X, y = cuml.make_blobs(n_samples, n_features, centers, cluster_std)
    columns = [f'feature{i+1}' for i in range(n_features)]
    df = cudf.DataFrame(X, columns=columns).astype('float32')
    df['label'] = y.astype(int)
    df.to_csv('dummy_data.csv', index=False)
    
    
def start_cluster():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    Comms.initialize(p2p=True)
    return cluster, client


def kill_cluster(cluster, client):
    Comms.destroy()
    client.close()
    cluster.close()
    
    
def compute_and_cache_knn_edgelist(input_csv_path, 
                                   knn_edgelist_path, 
                                   features, 
                                   n_neighbors,
                                   index_col = None,
                                   client=None):
    
    print(f'Computing and caching {n_neighbors}NN '
          f'edgelist: {knn_edgelist_path}')
    
    if client:
        chunksize = cugraph.dask.get_chunksize(input_csv_path)
        X = dask_cudf.read_csv(input_csv_path, chunksize=chunksize, index_col = index_col)
        X = X.loc[:, features].astype('float32')
        model = DaskNN(n_neighbors=n_neighbors+1, client=client)
    else:
        X = cudf.read_csv(input_csv_path, index_col = index_col)
        X = X.loc[:, features].astype('float32')
        model = NN(n_neighbors=n_neighbors+1)
    
    model.fit(X)
    
    n_vertices = X.shape[0].compute() if client else X.shape[0]
    
    # exclude self index
    knn_edgelist = model.kneighbors(X, return_distance=False).loc[:, 1:]  
    if client: # gather from GPUs and make index a contiguous range
        knn_edgelist = knn_edgelist.compute().reset_index(drop=True)
    knn_edgelist = knn_edgelist.melt(var_name='knn', value_name='dst')
    knn_edgelist = knn_edgelist.reset_index().rename(columns={'index':'src'})
    knn_edgelist = knn_edgelist.loc[:, ['src', 'dst']]
    knn_edgelist['src'] = knn_edgelist['src'] % n_vertices # avoids transpose
    knn_edgelist.to_parquet(knn_edgelist_path, index=(True if index_col is not None else False))
    
    
def compute_and_cache_jac_edgelist(knn_edgelist_path, 
                                   jac_edgelist_path, 
                                   distributed=False,
                                   index_col = None):
    
    print(f'Computing and caching jaccard edgelist: {jac_edgelist_path}')
    knn_graph = load_knn_graph(knn_edgelist_path, index_col, distributed)
    jac_graph = cugraph.jaccard(knn_graph)
    jac_graph.to_parquet(jac_edgelist_path, index=(True if index_col is not None else False))
    
    
def load_knn_graph(knn_edgelist_path, index_col=None, distributed=False):
    G = cugraph.Graph()
    if distributed:
        knn_edgelist = dask_cudf.read_parquet(knn_edgelist_path, 
                                              split_row_groups=True,
                                              index_col = index_col)
        G.from_dask_cudf_edgelist(knn_edgelist, source='src', destination='dst')
    else:
        knn_edgelist = cudf.read_parquet(knn_edgelist_path, index_col = index_col)
        G.from_cudf_edgelist(knn_edgelist, source='src', destination='dst')
    return G


def load_jac_graph(jac_edgelist_path, index_col = None, distributed=False):
    G = cugraph.Graph()
    if distributed:
        jac_edgelist = dask_cudf.read_parquet(jac_edgelist_path, 
                                              split_row_groups=True,
                                              index_col = index_col)
        G.from_dask_cudf_edgelist(jac_edgelist, edge_attr='jaccard_coeff')
    else:
        jac_edgelist = cudf.read_parquet(jac_edgelist_path, index_col = index_col)
        G.from_cudf_edgelist(jac_edgelist, edge_attr='jaccard_coeff')
    return G


def sort_by_size(clusters, min_size):
    """
    Relabel clustering in order of descending cluster size.
    New labels are consecutive integers beginning at 0
    Clusters that are smaller than min_size are assigned to -1.
    Copied from https://github.com/jacoblevine/PhenoGraph.
    
    Parameters
    ----------
    clusters: array
        Either numpy or cupy array of cluster labels.
    min_size: int
        Minimum cluster size.
    Returns
    -------
    relabeled: cupy array
        Array of cluster labels re-labeled by size.
        
    """
    relabeled = cp.zeros(clusters.shape, dtype=int)
    _, counts = cp.unique(clusters, return_counts=True)
    # sizes = cp.array([cp.sum(clusters == x) for x in cp.unique(clusters)])
    o = cp.argsort(counts)[::-1]
    for i, c in enumerate(o):
        if counts[c] > min_size:
            relabeled[clusters == c] = i
        else:
            relabeled[clusters == c] = -1
    return relabeled


################################################
# NOTES #
# Jaccard similarity and Leiden clustering don't
# have distributed GPU implementations yet,
# but they probably will soon, at which point
# it will be worth loading graphs using
# dask_cudf edgelists. As of RAPIDS 22.08 there
# is a distributed GPU implementation of Louvain
# if you run out of memory on single GPU
# computation of Leiden clustering. Note that
# such changes to RAPIDS will likely require
# reworking this code to accomodate, but should
# not be too much, e.g. change cugraph.jaccard
# to cugraph.dask.jaccard, etc.
################################################
def cluster(input_csv_path,
            features,
            data=None,
            n_neighbors=30,
            distributed_knn = True,
            distributed_graphs = False,
            index_col = None,
            new_column_name = 'cluster',
            overwrite = False,
            min_size=10):
    """
    CE 04/13/23: I added index_col so that if we save a sub-dataframe with indeces, 
    we can maintain the indeces from the indeces from the original dataframe. 
    I also added "new_column_name" so that we may further delineate sub-clusters within clusters. 
    Specifically, we could pass 'cluster' or 'subcluster' as an argument.
    Also added overwrite, in case we wish to entirely redo the clustering.
    
    It is important to note that 'features' are the only columns that will be used to cluster the cells.
    """
    
    tic = time.time()

    # client=None
    # if any([distributed_knn, distributed_graphs]):
    #     print('Initializing distributed GPU cluster...')
    #     cluster, client = start_cluster()
    #     print(f'Cluster started in {(time.time()-tic):.2f} seconds...')

    knn_edgelist_path = os.path.basename(input_csv_path).rsplit('.', 1)[0]
    knn_edgelist_path = f'{knn_edgelist_path}_{n_neighbors}NN_edgelist.parquet'

    jac_edgelist_path = os.path.basename(knn_edgelist_path).rsplit('.', 1)[0]
    jac_edgelist_path = f'{jac_edgelist_path}_jaccard.parquet'

    subtic = time.time()

    if os.path.exists(jac_edgelist_path) and (overwrite==False):

        print(f'Loading cached jaccard edgelist into graph: {jac_edgelist_path}')

        # if not distributed_graphs:
        #     kill_cluster(cluster, client)

        jac_graph = load_jac_graph(jac_edgelist_path,
                                   index_col,
                                   distributed_graphs)

        print(f'Jaccard graph loaded in {(time.time()-subtic):.2f} seconds...')

    elif os.path.exists(knn_edgelist_path) and (overwrite==False):

        print('Loading cached kNN edgelist for Jaccard graph '
              f'computation: {knn_edgelist_path}')

        # if not distributed_graphs:
        #     kill_cluster(cluster, client)

        compute_and_cache_jac_edgelist(knn_edgelist_path, 
                                       jac_edgelist_path, 
                                       distributed_graphs,
                                       index_col)

        jac_graph = load_jac_graph(jac_edgelist_path, 
                                   index_col,
                                   distributed_graphs)

        print('Jaccard graph computed, cached, and reloaded in '
              f'{(time.time()-subtic):.2f} seconds...')

    else:
        """
        The next 6 lines were commented out, but they enable multi-gpu support.
        Due to the driver version on our GPUs, this isn't working correctly.
        https://gitlab.com/eburling/grapheno/-/blob/master/grapheno/cluster.py
        """
        # with LocalCUDACluster() as cluster, Client(cluster) as client:
        #     compute_and_cache_knn_edgelist(input_csv_path, 
        #                                    knn_edgelist_path, 
        #                                    features, 
        #                                    n_neighbors, 
        #                                    client)
        compute_and_cache_knn_edgelist(input_csv_path, 
                                        knn_edgelist_path, 
                                        features, 
                                        n_neighbors,
                                        index_col)

        print(f'{n_neighbors}NN edgelist computed and cached in '
              f'{(time.time()-subtic):.2f} seconds...')

        subtic = time.time()

        # if not distributed_graphs:
        #     kill_cluster(cluster, client)

        compute_and_cache_jac_edgelist(knn_edgelist_path, 
                                       jac_edgelist_path, 
                                       distributed_graphs,
                                       index_col)

        jac_graph = load_jac_graph(jac_edgelist_path,
                                   index_col,
                                   distributed_graphs)

        print('Jaccard graph computed, cached, and reloaded in '
              f'{(time.time()-subtic):.2f} seconds...')

    subtic = time.time()

    print('Computing Leiden clustering over Jaccard graph...')
    clusters, modularity = cugraph.leiden(jac_graph)
    print(f'Leiden clustering completed in {(time.time()-subtic):.2f} seconds...')
    print(f'Clusters detected: {len(clusters.partition.unique())}')
    print(f'Clusters modularity: {modularity}')
        
    clusters = clusters.sort_values(by='vertex').partition.values
    clusters = sort_by_size(clusters, min_size)

    out_parquet_path = input_csv_path.rsplit('.', 1)[0]
    out_parquet_path = f'{out_parquet_path}_{n_neighbors}NN_leiden.parquet'
    print(f'Writing output dataframe: {out_parquet_path}')
    
    df = cudf.read_csv(input_csv_path, index_col = index_col)
    df[new_column_name] = clusters
    df.index.name = None #if we loaded from a csv file containing an index column, it autoloads a header name for it. Delete it.
    df.to_parquet(out_parquet_path, index=(True if index_col is not None else False))
    df = cudf.read_parquet(out_parquet_path, index_col = index_col)
    print(f'Grapheno completed in {(time.time()-tic):.2f} seconds!')
    
    return df