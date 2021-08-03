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
