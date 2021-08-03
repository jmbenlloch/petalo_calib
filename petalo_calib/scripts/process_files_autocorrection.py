from petalo_calib.tdc_corrections import correct_tfine_wrap_around
from petalo_calib.qdc_corrections import correct_efine_wrap_around

from petalo_calib.tdc_corrections import apply_tdc_correction
from petalo_calib.qdc_corrections import apply_qdc_autocorrection
from petalo_calib.qdc_corrections import apply_qdc_autocorrection

from petalo_calib.tdc_corrections import compute_integration_window_size
from petalo_calib.tdc_corrections import add_tcoarse_extended_to_df

from petalo_calib.clustering import compute_evt_number_combined_with_cluster_id
from petalo_calib.io         import compute_file_chunks_indices
from petalo_calib.io         import write_corrected_df_daq

import pandas as pd
import sys


def process_daq_df(df, df_tdc, df_qdc):
    compute_integration_window_size(df)

    correct_tfine_wrap_around(df)
    correct_efine_wrap_around(df)

    df = apply_tdc_correction(df, df_tdc)

    df['efine'] = apply_qdc_autocorrection(df, df_qdc)

    add_tcoarse_extended_to_df(df)

    df.drop(columns=['card_id', 'wordtype_id'], inplace=True)
    compute_evt_number_combined_with_cluster_id(df)
    return df



def process_daq_file(filein, fileout, df_tdc, qf_qdc):
    chunks = compute_file_chunks_indices(filein)
    nchunks = chunks.shape[0]

    for i in range(nchunks-1):
        print("{}/{}".format(i, nchunks-2))
        start = chunks[i]
        end   = chunks[i+1]

        df = pd.read_hdf(filein, 'data', start=start, stop=end+1)

        df_corrected = process_daq_df(df, df_tdc, df_qdc)
        write_corrected_df_daq(fileout, df_corrected, i, i>0)


efine_cal = '/home/jmbenlloch/calibration/corrections/scripts/autocorrection_5percent_10927.h5'
tfine_cal = '/home/jmbenlloch/calibration/corrections/scripts/asic0_tfine_cal.h5'

df_qdc = pd.read_hdf(efine_cal)
df_tdc = pd.read_hdf(tfine_cal,key='tfine_cal')

filein  = sys.argv[1]
fileout = sys.argv[2]
process_daq_file(filein, fileout, df_tdc, df_qdc)
