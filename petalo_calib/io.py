import pandas as pd
import tables as tb
import numpy  as np

from glob import glob


def compute_file_chunks_indices(filein):
    with tb.open_file(filein) as h5in:
        evt_numbers = h5in.root.data.cols.evt_number[:]
        evt_diffs  = np.diff(evt_numbers)
        evt_limits = np.where(evt_diffs)[0]

        # Find borders that keep ~chunk_size rows per chunk
        chunk_size   = 500000
        chunk_diffs  = np.diff(evt_limits // chunk_size)
        chunk_limits = np.where(chunk_diffs)[0]

        chunks = np.concatenate([np.array([0]),
                         evt_limits[chunk_limits],
                         np.array([evt_numbers.shape[0]])])
        return chunks


def write_corrected_df_daq(fileout, df, iteration, append=False):
    table_name = 'data_{}'.format(iteration)
    mode = 'a' if append else 'w'
    store = pd.HDFStore(fileout, mode, complib=str("zlib"), complevel=4)
    store.put(table_name, df, index=False, format='table', data_columns=None)
    store.close()



def get_files(run):
    folder = f'/analysis/{run}/hdf5/data/*h5'
    files = glob(folder)
    files.sort()
    return files


def read_run_data(files, verbose=False):
    dfs = []
    for i, fname in enumerate(files):
        if verbose:
            print(i, fname)
        try:
            df = pd.read_hdf(fname, 'data')
            df['fileno'] = i
            dfs.append(df)
        except:
            print("Error in file ", fname)
    df = pd.concat(dfs).reset_index(drop=True)
    return df


def get_evt_times(files, verbose=False):
    time_dfs = []

    for i, fname in enumerate(files):
        if verbose:
            print(i, fname)
        df_time = pd.read_hdf(fname, 'dateEvents')
        df_time['fileno'] = i
        time_dfs.append(df_time)
    df_times = pd.concat(time_dfs)
    df_times['date'] = pd.to_datetime(df_times.timestamp, unit='us')

    # Compute time difference between one event and the next
    df_times['time_diff'] = np.abs((df_times.timestamp/1e6).diff(periods=-1))
    df_times = df_times.fillna(0)
    return df_times
