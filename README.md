# PETALOB_CALIB README

This repository is a preliminary software to process runs from PETALO. The software is capable of creating one job per file in a given run and sends them to the processing queue. Given the memory limitations, each file is processed by chunks (~500k rows each).

# How to use

Clone the repository and run:

```bash
source setup.sh
```

Running the script without arguments shows the options:

```bash
$ python petalo_calib/process_run.py 
Missing arguments. Usage:

          python process_run.py [run_number] [folder_name] [script]

    Where:
      - run_number: run to be processed
      - folder_name: name of the folder containing the output files to be
        created at /analysis/[run_number]/hdf5/proc/[folder_name]
      - script_to_run: script to be run from $PETALO_CALIB/scripts/[script_to_run]
```

An example would be:

```bash
python petalo_calib/process_run.py 10927 linear_interp process_files_linear_interpolation.py
```

That command will send a job for each file in run 10927 using the processing script `petalo_calib/process_files_linear_interpolation.py`.

The relevant files are created in:

```bash
/analysis/10927/hdf5/proc/linear_interp/files
/analysis/10927/hdf5/proc/linear_interp/jobs
/analysis/10927/hdf5/proc/linear_interp/logs
```

If a different script is used for the same run, you **must** change the name (`linear_interp` in the example). Otherwise, all the files would be overwritten.

# Calibration procedures

The software can be extended with new calibration procedures to subtract the intrinsic polarization current of the ASIC. In order to do so, a new script should be added to `petalo_calib/petalo_calib/scripts/`. Currently there are to options there, but the idea to create a new one is fairly simple.

As an example, here is the commented code of the linear interpolation version:

```python
from petalo_calib.tdc_corrections import correct_tfine_wrap_around
from petalo_calib.qdc_corrections import correct_efine_wrap_around

from petalo_calib.tdc_corrections import apply_tdc_correction
from petalo_calib.qdc_corrections import compute_efine_correction_using_linear_interpolation
from petalo_calib.qdc_corrections import create_qdc_interpolator_df

from petalo_calib.tdc_corrections import compute_integration_window_size
from petalo_calib.tdc_corrections import add_tcoarse_extended_to_df

from petalo_calib.clustering import compute_evt_number_combined_with_cluster_id
from petalo_calib.io         import compute_file_chunks_indices
from petalo_calib.io         import write_corrected_df_daq

import pandas as pd
import sys

# This is the function that corrects each chunk of the file.
# Should be adapted to call the functions you need.
def process_daq_df(df, df_tdc, df_qdc):
    compute_integration_window_size(df)

    correct_tfine_wrap_around(df)
    correct_efine_wrap_around(df)

    df = apply_tdc_correction(df, df_tdc)

    compute_efine_correction_using_linear_interpolation(df, df_qdc)

    add_tcoarse_extended_to_df(df)

    df.drop(columns=['card_id', 'wordtype_id'], inplace=True)
    compute_evt_number_combined_with_cluster_id(df)
    return df

# This function reads the HDF5 input file by chunks and
# calls the processing function for each of them
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

# Do whatever you need to read your calibration files
qdc0_fname = '/home/jmbenlloch/calibration/corrections/qdc_asic0_10772_linear_interpolation.h5'
qdc2_fname = '/home/jmbenlloch/calibration/corrections/qdc_asic2_10746_linear_interpolation.h5'
df_qdc     = create_qdc_interpolator_df(qdc0_fname, qdc2_fname)

tfine_cal = '/home/jmbenlloch/calibration/corrections/scripts/asic0_tfine_cal.h5'
df_tdc = pd.read_hdf(tfine_cal,key='tfine_cal')

# Call the function with the input & output files and the calibration data to be used.
filein  = sys.argv[1]
fileout = sys.argv[2]
process_daq_file(filein, fileout, df_tdc, df_qdc)
```

To create a new version you should add the relevant functions to:

`petalo_calib/tdc_corrections.py` and `petalo_calib/qdc_corrections.py`. Then you can create a copy of `petalo_calib/petalo_calib/scripts/process_files_linear_interpolation.py` and make the proper changes to use your correction functions.

## Output files

The output files will not have only one big dataframe with all the data together Instead, you will find different tables named from `data0` to `dataN` where N is the number of chunks for each file.

### IMPORTANT: Cluster and event IDs

To avoid overflows, the cluster ID is reset for each `evt_number` . To process individual events you have to filter by both `evt_number` and  `cluster`. 

In `petalo_calib/examples/compute_spectrums.py` is shown how to read the output files.
