import numpy as np
from sklearn.cluster import DBSCAN


def compute_clusters(df):
    values = df.tcoarse_extended.values
    values = values.reshape(values.shape[0],1)

    clusters = DBSCAN(eps=10, min_samples=2).fit(values)
    return clusters.labels_


def compute_evt_number_combined_with_cluster_id(df):
    '''
    Assign a cluster id for each real event. The set of cluster ids are independent
    for each DATE event to avoid overflows. To filter individual events both 
    'evt_number' and 'cluster' have to be used.
    '''
    df_clusters = df.groupby('evt_number').apply(compute_clusters)
    df['cluster'] = np.concatenate(df_clusters.values)
#    df.loc[df[df.cluster != -1].index, 'cluster'] = df.evt_number * 20000 + df.cluster
    df.loc[df[df.cluster != -1].index, 'cluster'] = df.cluster
