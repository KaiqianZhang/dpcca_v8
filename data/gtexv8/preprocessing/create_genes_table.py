"""=============================================================================
Script to create and save Pandas dataframe of each gene.
============================================================================="""

import numpy as np
import pandas as pd
import os

# ------------------------------------------------------------------------------

genes = set()
for f in next(os.walk('genes'))[-1]:
    df    = pd.read_csv('genes/%s' % f, delimiter='\t')
    gs    = df['ID'].tolist()
    gs    = set(gs)
    genes = genes.union(gs)
    print('%s\t%s' % (len(genes), f))

np.save('genes.npy', genes)
