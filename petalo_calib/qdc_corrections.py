import pandas as pd
import numpy  as np

from scipy.interpolate import interp1d


def correct_efine_wrap_around(df):
    df['efine'] = (df['efine'] + 14) % 1024


# Autocorrection
def apply_qdc_autocorrection(df, df_qdc):
    df = df.merge(df_qdc, on=['tofpet_id', 'channel_id', 'tac_id', 'intg_w'])
    return (df['efine'] - df.offset)


# Linear interpolation
def compute_qdc_calibration_using_mode(fname_qdc):
    dfs = []
    for channel in range(64):
        #print(channel)
        df_tpulse = pd.read_hdf(fname_qdc, f'ch{channel}')

        compute_integration_window_size(df_tpulse)
        df_tpulse['efine'] = (df_tpulse['efine'] + 14) % 1024

        df_calib = df_tpulse.groupby(['tofpet_id', 'channel_id', 'tac_id', 'intg_w'])['efine'].agg(lambda x: x.value_counts().index[0])
        df_calib = df_calib.reset_index()
        dfs.append(df_calib)

    df_calib = pd.concat(dfs)
    df_calib.reset_index(drop=True, inplace=True)

    # Replace first entry (3) with 0 to avoid out of range errors when interpolating (really) small windows.
    df_calib.loc[df_calib.intg_w == 3, 'intg_w'] = 0
    # Replace last entry (291) with 2000 to avoid out of range errors when interpolating (really) large windows.
    df_calib.loc[df_calib.intg_w == 291, 'intg_w'] = 2000

    return df_calib


def create_qdc_interpolator_df(fname_qdc_0, fname_qdc_2):
    df_qdc0 = pd.read_hdf(fname_qdc_0)
    df_qdc2 = pd.read_hdf(fname_qdc_2)
    df_qdc = pd.concat([df_qdc0, df_qdc2]).reset_index()
    df_interpolators = df_qdc.groupby(['tofpet_id', 'channel_id', 'tac_id']).apply(lambda df: interp1d(df.intg_w, df.efine))
    return df_interpolators


def compute_efine_correction_using_linear_interpolation(df, df_interpolators):
    df['correction']      = df.apply(lambda row: df_interpolators[row.tofpet_id, row.channel_id, row.tac_id](row.intg_w), axis=1)
    df['efine_corrected'] = (df.efine - df.correction).astype(np.float64) # For some reason df.correction is Object, not f64
    df.drop(columns=['correction'], inplace=True)
