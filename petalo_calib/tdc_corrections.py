import numpy as np


def correct_tfine_wrap_around(df):
    df['tfine'] = (df['tfine'] + 14) % 1024


def compute_tcoarse_wrap_arounds(df):
    limits = df[df.tcoarse_diff < -30000].index
    first  = df.index[0]
    last   = df.index[-1]
    limits = np.concatenate([np.array([first]), limits.values, np.array([last])])
    return limits


def compute_tcoarse_nloops_per_event(df):
    limits = df.groupby('evt_number').apply(compute_tcoarse_wrap_arounds)

    nloops = np.zeros(df.shape[0], dtype='int32')

    for evt_limits in limits.values:
        for i in range(evt_limits.shape[0]-1):
            start = evt_limits[i]
            end   = evt_limits[i+1]

            nloops[start:end+1] = i

    return nloops


def compute_extended_tcoarse(df):
    return df['tcoarse'] + df['nloops'] * 2**16


def add_tcoarse_extended_to_df(df):
    df['tcoarse']          = df.tcoarse.astype(np.int32)
    df['tcoarse_diff']     = df.tcoarse.diff()
    df['nloops']           = compute_tcoarse_nloops_per_event(df)
    df['tcoarse_extended'] = compute_extended_tcoarse(df)
    df.drop(columns=['tcoarse_diff', 'nloops'], inplace=True)


def compute_integration_window_size(df):
    df['intg_w'] = (df.ecoarse - (df.tcoarse % 2**10)).astype('int16')
    df.loc[df['intg_w'] < 0, 'intg_w'] += 2**10


def apply_tdc_correction(df, df_tdc):
    df = df.reset_index().merge(df_tdc[['channel_id', 'tac_id', 'amplitude', 'offset']], on=['channel_id', 'tac_id'])
    df = df.sort_values('index').set_index('index')
    df.index.name = None

    period = 360
    df['tfine_corrected'] = (period/np.pi)*np.arctan(1/np.tan((np.pi/(-2*df.amplitude))*(df.tfine-df.offset)))
    df.loc[df['tfine_corrected'] < 0, 'tfine_corrected'] += period
    df = df.drop(columns=['amplitude', 'offset'])
    df['t'] = df.tcoarse - (360 - df.tfine_corrected) / 360
    return df
