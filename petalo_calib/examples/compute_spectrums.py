import numpy as np
import tables as tb
import pandas as pd

import sys


def filter_evt_with_max_charge_at_center(df):
    argmax = df['efine'].argmax()
    return df.iloc[0].sensor_id in [44, 45, 54, 55]


def compute_energy_spectrum(df, bins=15000, range=[0, 15000]):
    energies = df.groupby(['evt_number', 'cluster'])['efine'].sum()
    counts, _ = np.histogram(energies, bins=bins, range=range)
    return counts


def compute_coincidences(df):
    nplanes = df.groupby(['evt_number', 'cluster'])['tofpet_id'].nunique()
    df_idx  = df.set_index(['evt_number', 'cluster'])
    df_coincidences = df_idx.loc[nplanes[nplanes == 2].index]
    return df_coincidences


def select_evts_with_max_charge_at_center(df):
    df_filter_center = df.groupby(['evt_number', 'cluster']).apply(filter_evt_with_max_charge_at_center)
    return df[df_filter_center]


def write_histograms_file(hist_all, hist_coincidences, hist_center, fileout):
    h5in = tb.open_file(fileout, mode = "w")
    table = h5in.create_array(h5in.root, 'hist_all', hist_all, "data")
    table.flush()

    table = h5in.create_array(h5in.root, 'hist_coincidences', hist_coincidences, "data")
    table.flush()

    table = h5in.create_array(h5in.root, 'hist_center', hist_center, "data")
    table.flush()
    h5in.close()


def process_file(filein, fileout):
    bins = 15000
    hist_all          = np.zeros(bins)
    hist_coincidences = np.zeros(bins)
    hist_center       = np.zeros(bins)
    xs = np.arange(bins) + 0.5

    store = pd.HDFStore(filein, 'r')

    for key in store.keys():
        print(key)
        df = store.get(key)
        df_clusters     = df[df.cluster != -1]
        df_coincidences = compute_coincidences(df_clusters)
        df_center       = select_evts_with_max_charge_at_center(df_coincidences)

        energies_all          = compute_energy_spectrum(df_clusters)
        energies_coincidences = compute_energy_spectrum(df_coincidences)
        energies_center       = compute_energy_spectrum(df_center)

        hist_all          += energies_all
        hist_coincidences += energies_coincidences
        hist_center       += energies_center

    write_histograms_file(hist_all, hist_coincidences, hist_center, fileout)



filein  = sys.argv[1]
fileout = sys.argv[2]
process_file(filein, fileout)
