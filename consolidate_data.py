# consolidate_data.py

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import sys

intensities_size = 1738
num_samples = 6000

if len(sys.argv) != 3:
    print ("syntax is: python consolidate_data.py data_folder output.csv")
    exit ()

DATA_PATH = Path(sys.argv[1])
OUTPUT_CSV = Path(sys.argv[2])

print ('input folder:', DATA_PATH) 
print ('output CSV  :', OUTPUT_CSV)

paths = list(DATA_PATH.iterdir())
random.shuffle(paths)
T = np.zeros ((num_samples, intensities_size))
ct = 1;
print ("\nconsolidating...")
for path in paths:
    df = pd.read_csv(path, names=['shift', 'intensity'])
    shifts = df['shift']
    intensities = df['intensity']
    T[ct] = intensities
    if ct % 1000 == 0:
         print (ct)
    ct += 1
    if ct >= num_samples:
        break
print('final shape:', T.shape)
print ('\nsaving...')
np.savetxt(OUTPUT_CSV, T, delimiter=',')

# eof
