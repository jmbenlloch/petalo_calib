from petalo_calib.tdc_corrections import correct_tfine_wrap_around
from petalo_calib.qdc_corrections import correct_efine_wrap_around

from petalo_calib.tdc_corrections import apply_tdc_correction_tot

from petalo_calib.tdc_corrections import compute_integration_window_size
from petalo_calib.tdc_corrections import add_tcoarse_extended_to_df

from petalo_calib.clustering import compute_evt_number_combined_with_cluster_id
from petalo_calib.io         import compute_file_chunks_indices
from petalo_calib.io         import write_corrected_df_daq

from sklearn.cluster import DBSCAN

import pandas as pd
import numpy  as np
import sys


def compute_tcoarse_wrap_arounds(df):
    limits = df[df.tcoarse_diff < -20000].index
    first  = df.index[0]
    last   = df.index[-1]
    limits = np.concatenate([np.array([first]), limits.values, np.array([last])])
    return limits


def compute_tcoarse_nloops(df):
    limits = compute_tcoarse_wrap_arounds(df)
    nloops = np.zeros(df.shape[0], dtype='int32')

    for i in range(limits.shape[0]-1):
        start = limits[i]
        end   = limits[i+1]

        nloops[start:end+1] = i

    return nloops


def compute_extended_tcoarse(df):
    return df['tcoarse'] + df['nloops'] * 2**16


def add_tcoarse_extended_to_df(df):
    df['tcoarse']          = df.tcoarse.astype(np.int32)
    df['tcoarse_diff']     = df.tcoarse.diff()
    df['nloops']           = compute_tcoarse_nloops(df)
    df['tcoarse_extended'] = compute_extended_tcoarse(df)


def local_sort_tcoarse(df, indices):
    start = -1
    end   = -1
    window_size = 120

    for index in indices:
        if (index >= start) and (index <= end):
            #print("Done! ", index)
            continue
        start = index - window_size
        end   = index + window_size
        #print(start, end)

        df.iloc[start:end] = df.iloc[start:end].sort_values('tcoarse', ascending=False)


def local_sort_tcoarse_to_fix_wrap_arounds(df):
    add_tcoarse_extended_to_df(df)
    indices = df[df.tcoarse_diff < -20000].index.values
    local_sort_tcoarse(df, indices)
    add_tcoarse_extended_to_df(df)
    #df.drop(columns=['tcoarse_diff', 'nloops'], inplace=True)


def compute_tcoarse_extended_with_local_sort(df):
    df_0 = df[df.tofpet_id == 0].reset_index()
    df_2 = df[df.tofpet_id == 2].reset_index()

    local_sort_tcoarse_to_fix_wrap_arounds(df_0)
    local_sort_tcoarse_to_fix_wrap_arounds(df_2)

    df_all = pd.concat([df_0, df_2])
    df_all_sorted = df_all.sort_values(['evt_number', 'tcoarse_extended']).reset_index(drop=True)
    return df_all_sorted


def compute_clusters(df):
    values = df.tcoarse_extended.values
    values = values.reshape(values.shape[0],1)

    clusters = DBSCAN(eps=10, min_samples=2).fit(values)
    return clusters.labels_


def process_daq_df_tot(df, df_tdc1_asic0, df_tdc2_asic0, df_tdc1_asic2, df_tdc2_asic2):
    compute_integration_window_size(df)

    correct_tfine_wrap_around(df)
    correct_efine_wrap_around(df)

    df = compute_tcoarse_extended_with_local_sort(df)

    df_0 = df[df.tofpet_id == 0]
    df_2 = df[df.tofpet_id == 2]

    df_0 = apply_tdc_correction_tot(df_0, df_tdc1_asic0, 'tfine')
    df_0 = apply_tdc_correction_tot(df_0, df_tdc2_asic0, 'efine')

    df_2 = apply_tdc_correction_tot(df_2, df_tdc1_asic2, 'tfine')
    df_2 = apply_tdc_correction_tot(df_2, df_tdc2_asic2, 'efine')

    df = pd.concat([df_0, df_2]).sort_index()

    df.drop(columns=['card_id', 'wordtype_id'], inplace=True)
    df['cluster'] = compute_clusters(df)
    return df



def process_daq_file(filein, fileout, df_tdc1_asic0, df_tdc2_asic0, df_tdc1_asic2, df_tdc2_asic2):
    chunks = compute_file_chunks_indices(filein)
    nchunks = chunks.shape[0]

    for i in range(nchunks-1):
        print("{}/{}".format(i, nchunks-2))
        start = chunks[i]
        end   = chunks[i+1]

        df = pd.read_hdf(filein, 'data', start=start, stop=end+1)

        df_corrected = process_daq_df_tot(df, df_tdc1_asic0, df_tdc2_asic0, df_tdc1_asic2, df_tdc2_asic2)
        write_corrected_df_daq(fileout, df_corrected, i, i>0)


tdc1_asic0 = '/home/vherrero/CALIBRATION_FILES/tfine_cal_asic0_run11291.h5'
tdc2_asic0 = '/home/vherrero/CALIBRATION_FILES/tfine2_cal_asic0_run11291.h5'

tdc1_asic2 = '/home/vherrero/CALIBRATION_FILES/tfine_cal_asic2_run11292.h5'
tdc2_asic2 = '/home/vherrero/CALIBRATION_FILES/tfine2_cal_asic2_run11292.h5'

df_tdc1_asic0 = pd.read_hdf(tdc1_asic0, key='tfine_cal')
df_tdc2_asic0 = pd.read_hdf(tdc2_asic0, key='tfine_cal')

df_tdc1_asic2 = pd.read_hdf(tdc1_asic2, key='tfine_cal')
df_tdc2_asic2 = pd.read_hdf(tdc2_asic2, key='tfine_cal')


filein  = sys.argv[1]
fileout = sys.argv[2]
process_daq_file(filein, fileout, df_tdc1_asic0, df_tdc2_asic0, df_tdc1_asic2, df_tdc2_asic2)
