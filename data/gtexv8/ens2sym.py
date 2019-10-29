"""=============================================================================
Convert ensemble IDs to gene symbols.
============================================================================="""

import mygene
from   data import GTExV8Config, GTExV8Dataset

# ------------------------------------------------------------------------------

cfg     = GTExV8Config()
dataset = GTExV8Dataset(cfg)
mg      = mygene.MyGeneInfo()
ens_ids = []

for ens in dataset.genes_df.index.values:
    ens_ids.append(ens.split('.')[0])

ginfo   = mg.querymany(ens_ids, scopes='ensembl.gene')
import pdb; pdb.set_trace()
