from petalo_calib.tdc_corrections import correct_tfine_wrap_around
from petalo_calib.qdc_corrections import correct_efine_wrap_around

from petalo_calib.tdc_corrections import apply_tdc_correction_tot

from petalo_calib.tdc_corrections import compute_integration_window_size
from petalo_calib.tdc_corrections import add_tcoarse_extended_to_df

from petalo_calib.clustering import compute_evt_number_combined_with_cluster_id
from petalo_calib.io         import compute_file_chunks_indices
from petalo_calib.io         import write_corrected_df_daq

import pandas as pd
import sys


def process_daq_df_tot(df, df_tdc1_asic0, df_tdc2_asic0, df_tdc1_asic2, df_tdc2_asic2):
    compute_integration_window_size(df)

    correct_tfine_wrap_around(df)
    correct_efine_wrap_around(df)

    add_tcoarse_extended_to_df(df)

    df_0 = df[df.tofpet_id == 0]
    df_2 = df[df.tofpet_id == 2]

    df = apply_tdc_correction(df, df_tdc)

    df_0 = apply_tdc_correction_tot(df_0, df_tdc1_asic0, 'tfine')
    df_0 = apply_tdc_correction_tot(df_0, df_tdc2_asic0, 'efine')

    df_2 = apply_tdc_correction_tot(df_2, df_tdc1_asic2, 'tfine')
    df_2 = apply_tdc_correction_tot(df_2, df_tdc2_asic2, 'efine')

    df = pd.concat([df_0, df_2]).sort_index()

    df.drop(columns=['card_id', 'wordtype_id'], inplace=True)
    compute_evt_number_combined_with_cluster_id(df)
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
