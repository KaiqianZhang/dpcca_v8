"""=============================================================================
Utility script to align gene expression level samples with images.
============================================================================="""

import csv
import glob
import numpy as np
import pandas as pd
import os
import random
import re
import sys
import torch

from   smtsdmap import SMTSD_TO_GENE_TSV

# ------------------------------------------------------------------------------

def main():
    df = pd.read_csv('sample_table.txt', delimiter='\t')

    all_genes = list(np.load('genes.npy').tolist())
    genes_idx = pd.Index(all_genes)

    total     = 0
    nomap     = 0
    nofindimg = 0
    nofindgen = 0

    samples = []
    genes   = []
    tissues = {}
    images_paths = {}

    for i, (samp_id, tissue) in enumerate(zip(df['SAMPID'], df['SMTSD'])):

        total  += 1
        parts   = samp_id.split('-')
        subj_id = '-'.join(parts[:2])
        tiss_id = parts[2]

        if type(tissue) is not str or not SMTSD_TO_GENE_TSV[tissue]:
            if 'SMTS' in df.iloc[i]:
                assert type(df.iloc[i]['SMTS']) is not str
            nomap += 1
            continue

        # Look for sample image.
        # ======================================================================
        image = get_image(subj_id, tiss_id, tissue)
        if image is None:
            nofindimg += 1
            continue

        # Look for sample genes.
        # ======================================================================
        base = SMTSD_TO_GENE_TSV[tissue]
        fname = '/tigress/gwg3/new_GTEX/genes/'\
                '%s.v8.normalized_expression.tsv' % base
        gexp = get_genes(subj_id, fname)
        if gexp is None:
            nofindgen += 1
            continue
        else:
            gexp = gexp.reindex(genes_idx, fill_value=0)
            assert len(genes_idx) == gexp.shape[0]
            assert not (gexp == 0).all().bool()
            assert gexp.columns[0] == subj_id
            full_id = '%s-%s' % (subj_id, tiss_id)
            # Ensure column names exactly match file names (sans '.png').
            gexp = gexp.rename(columns={subj_id: full_id})
    

        # Pair samples.
        # ======================================================================
        print('Matched: %s, %s' % (image.split('/')[-1].split('.')[0],
                                   gexp.columns[0]))
        sample = '-'.join(samp_id.split('-')[:3])
        samples.append(sample)
        genes.append(gexp)
        assert full_id not in images_paths
        assert full_id not in tissues
        # This mapping might seem redundant, but it is not. For example, the
        # full_id could be: 'GTEX-1117F-0226' while the image name would be
        # 'GTEX-1117F-0225.png'. Note the last digit is changed. So just
        # appending '.png' is insufficient.
        images_paths[full_id] = image
        tissues[full_id] = tissue

    print('=' * 80)
    print('Total               : %s' % total)
    print('No mapping          : %s' % nomap)
    print('Could not find image: %s' % nofindimg)
    print('Could not find genes: %s' % nofindgen)
    lost = nomap + nofindimg + nofindgen
    num_samps = total - lost
    assert num_samps == len(samples) == len(images_paths) == len(genes)
    print('New dataset         : %s' % num_samps)

    print('=' * 80)
    print('Creating genes data frame')

    genes_df   = pd.concat(genes, axis=1, sort=True)
    images     = {k:v.split('/')[-1] for k,v in images_paths.items()}
    images_df  = pd.DataFrame.from_dict(images, orient='index',
                                        columns=['filename'])
    tissues_df = pd.DataFrame.from_dict(tissues, orient='index',
                                        columns=['tissue'])
    torch.save({
        'samples': samples,
        'images' : images_df,
        'genes'  : genes_df,
        'tissues': tissues_df
    }, 'train.pth')

    # Move images to new location.
    '''print('Moving images.')
    for i, (full_id, impath) in enumerate(images_paths.items()):
        cmd = 'cp %s /scratch/gpfs/gwg3/dmcm/data/gtexv8/images/' % impath
        os.system(cmd)
        if i % 100 == 0:
            print(cmd)
            print('%s / %s' % (i, len(images)))
    '''

# ------------------------------------------------------------------------------

def get_image(subj_id, tiss_id, expected_tissue):
    root = '/tigress/gwg3/new_GTEX/images'
    matches = []
    for directory in next(os.walk(root))[1]:
        if directory == root:
            continue

        assert len(tiss_id) == 4
        leading_zero = tiss_id.startswith('0')

        for id_ in range(int(tiss_id) - 1, int(tiss_id) + 2):
            if leading_zero:
                id_ = str(id_).zfill(4)
            fname = '%s/%s/%s-%s.png' % (root, directory, subj_id, id_)
            match = glob.glob(fname)
            matches += match

    if len(matches) > 1:
        return verify_multiple_matches(subj_id, tiss_id, matches)
    if len(matches) == 1:
        return matches[0]
    return None

# ------------------------------------------------------------------------------

# Loading TSVs is slow. Just cache for a faster script.
loaded_dfs = {}

def get_genes(subj_id, fname):
    if fname not in loaded_dfs:
        loaded_dfs[fname] = pd.read_csv(fname, delimiter='\t')
    df = loaded_dfs[fname]
    try:
        df = df[['ID', subj_id]].set_index('ID')
    except KeyError:
        return None
    return df

# ------------------------------------------------------------------------------

def verify_multiple_matches(subj_id, tiss_id, matches):
    first_samp = matches[0]
    first_tiss = first_samp.split('/')[5]
    first_last_num = first_samp.split('/')[-1].split('-')[-1][:-4]
    for match in matches:
        this_subj_id = '-'.join(match.split('/')[6].split('-')[:2])
        assert subj_id == this_subj_id
        # If this assert fails, I have a major misunderstanding in how the data
        # are organized.
        tiss = match.split('/')[5]
        assert tiss == first_tiss
        last_num = match.split('/')[-1].split('-')[-1][:-4]
        assert np.abs(int(first_last_num) - int(last_num)) <= 1
    # If the asserts pass, the samples are equivalent.
    return matches[0]

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
