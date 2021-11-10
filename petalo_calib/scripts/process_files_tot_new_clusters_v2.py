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


def compute_nloops(df, limits):
    nloops = np.zeros(df.shape[0], dtype='int32')

    first  = df.index[0]
    last   = df.index[-1]
    limits = np.concatenate([np.array([first]), limits.values, np.array([last])])

    for i in range(limits.shape[0]-1):

        start = limits[i  ]
        end   = limits[i+1]

        nloops[start:end+1] = i

    return nloops


def sort_tcoarse(df):
    while True:
        df['tcoarse']      = df.tcoarse.astype(np.int32)
        df['tcoarse_diff'] = df.tcoarse.diff()

        limits       = df[df.tcoarse_diff < -30000].index
        to_be_sorted = df[df.tcoarse_diff >  30000].index
        #print(limits)
        #print(to_be_sorted)

        if len(to_be_sorted) == 0:
            print("Already sorted!")
            break

        position = to_be_sorted[0]

        # Compute closest limit to "position"
        try:
            last_limit_index = np.where((limits - position) < 0)[0][-1]
            last_limit       = limits[last_limit_index] -1
            print("Position: {}\t limit: {}".format(position, last_limit))
        except IndexError:
            print("Dropped row {}".format(df.loc[position].old_index))
            df = df.drop(position)
            continue

        diffs       = df.loc[position+1:].tcoarse_diff
        n_elements  = np.where(diffs < -30000)[0][0]
        upper_limit = position + n_elements

        chunk1 = df.loc[:last_limit]
        chunk2 = df.loc[position:upper_limit]
        chunk3 = df.loc[last_limit+1:position-1]
        chunk4 = df.loc[upper_limit+1:]

        df = pd.concat([chunk1, chunk2, chunk3, chunk4]).reset_index(drop=True)
        #print("\n\n\n\n")
    return df, limits


def compute_extended_tcoarse(df):
    return df['tcoarse'] + df['nloops'] * 2**16


def compute_tcoarse_extended_with_local_sort(df, nloop_offsets):
    df_0 = df[df.tofpet_id == 0].reset_index()
    df_2 = df[df.tofpet_id == 2].reset_index()

    # debug
    df_0['old_index'] = df_0.index
    df_2['old_index'] = df_2.index

    df_0, limits_0 = sort_tcoarse(df_0)
    df_2, limits_2 = sort_tcoarse(df_2)

    df_0['nloops'] = compute_nloops(df_0, limits_0) + nloop_offsets[0]
    df_2['nloops'] = compute_nloops(df_2, limits_2) + nloop_offsets[1]

    df_0['tcoarse_extended'] = compute_extended_tcoarse(df_0)
    df_2['tcoarse_extended'] = compute_extended_tcoarse(df_2)

    df_all        = pd.concat([df_0, df_2])
    df_all_sorted = df_all.sort_values(['evt_number', 'tcoarse_extended']).reset_index(drop=True)

    nloop_ends = [df_0.nloops.max(), df_2.nloops.max()]

    return df_all_sorted, nloop_ends


def compute_clusters(df):
    values = df.tcoarse_extended.values
    values = values.reshape(values.shape[0],1)

    clusters = DBSCAN(eps=10, min_samples=2).fit(values)
    return clusters.labels_


def process_daq_df_tot(df, df_tdc1_asic0, df_tdc2_asic0, df_tdc1_asic2, df_tdc2_asic2, nloop_offsets):
    compute_integration_window_size(df)

    correct_tfine_wrap_around(df)
    correct_efine_wrap_around(df)

    df, nloop_ends = compute_tcoarse_extended_with_local_sort(df, nloop_offsets)

    df_0 = df[df.tofpet_id == 0]
    df_2 = df[df.tofpet_id == 2]

    df_0 = apply_tdc_correction_tot(df_0, df_tdc1_asic0, 'tfine')
    df_0 = apply_tdc_correction_tot(df_0, df_tdc2_asic0, 'efine')

    df_2 = apply_tdc_correction_tot(df_2, df_tdc1_asic2, 'tfine')
    df_2 = apply_tdc_correction_tot(df_2, df_tdc2_asic2, 'efine')

    df = pd.concat([df_0, df_2]).sort_index()

    df.drop(columns=['card_id', 'wordtype_id'], inplace=True)
    df['cluster'] = compute_clusters(df)
    return df, nloop_ends


def process_daq_file(filein, fileout, df_tdc1_asic0, df_tdc2_asic0, df_tdc1_asic2, df_tdc2_asic2):
    chunks = compute_file_chunks_indices(filein)
    nchunks = chunks.shape[0]

    nloop_offsets = [0, 0]

    for i in range(nchunks-1):
        print("{}/{}".format(i, nchunks-2))
        start = chunks[i]
        end   = chunks[i+1]

        df = pd.read_hdf(filein, 'data', start=start, stop=end+1)

        df_corrected, nloop_offsets = process_daq_df_tot(df, df_tdc1_asic0, df_tdc2_asic0, df_tdc1_asic2, df_tdc2_asic2, nloop_offsets)
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
