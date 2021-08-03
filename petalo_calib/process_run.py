from petalo_calib.io import get_files

from subprocess import call

import sys
import os


if len(sys.argv) < 3:
    print('''Missing arguments. Usage:

          python process_run.py [run_number] [folder_name] [script]

    Where:
      - run_number: run to be processed
      - folder_name: name of the folder containing the output files to be
        created at /analysis/[run_number]/hdf5/proc/[folder_name]
      - script_to_run: script to be run from $PETALO_CALIB/scripts/[script_to_run]
''')
    exit(0)

run_number    = sys.argv[1]
folder_name   = sys.argv[2]
script_to_run = sys.argv[3]


sw_path     = os.environ['PETALO_CALIB']
script_path = os.path.join(sw_path, 'petalo_calib/scripts', script_to_run)

base_path = f'/analysis/{run_number}/hdf5/proc/{folder_name}/'
out_path  = os.path.join(base_path, 'files')
job_path  = os.path.join(base_path, 'jobs')
log_path  = os.path.join(base_path, 'logs')

os.makedirs(job_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)
os.makedirs(out_path, exist_ok=True)



template ='''#PBS -N {name}
#PBS -q medium
#PBS -o /dev/null
#PBS -e /dev/null
source /software/miniconda/etc/profile.d/conda.sh
conda activate rawdata
export PYTHONPATH={sw_path}:$PYTHONPATH
python {script} {filein} {fileout} 1>{logout} 2>{logerr}
'''

files = get_files(run_number)

for fname in files:
    print(fname)

    basename = fname.split('/')[-1]
    fnumber = basename.split('_')[2]

    params = {
            'name'    : '{}_{}_{}'.format(folder_name, run_number, fnumber),
            'filein'  : fname,
            'fileout' : os.path.join(out_path, basename),
            'logout'  : os.path.join(log_path, basename + '.out'),
            'logerr'  : os.path.join(log_path, basename + '.err'),
            'script'  : script_path,
            'sw_path' : sw_path,
    }

    job_fname = '{}/{}.sh'.format(job_path, basename)
    with open(job_fname, 'w') as jobfile:
        jobfile.write(template.format(**params))

    call(f'qsub {job_fname}', shell=True)

