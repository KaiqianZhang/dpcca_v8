"""=============================================================================
Utility script to compare the contents of two directories before and after
processing a tissue.
============================================================================="""

import os
import sys

# ------------------------------------------------------------------------------

def main():
    tissue = sys.argv[1]
    source_dir = '/tigress/gwg3/Tissues/%s' % tissue
    target_dir = '/tigress/gwg3/new_GTEX/%s' % tissue
    
    source_files = []
    target_files = []

    for fname in os.listdir(source_dir):
        if not fname.endswith('.svs'):
            if not fname.endswith('.svs\r'):
                raise AttributeError('Not Aperio image file.')
            else:
                files = [f for f in os.listdir(source_dir)]
                assert fname.replace('\r', '') in files
                continue
        fname = fname.replace('.svs', '')
        source_files.append(fname)
    source_files.sort()

    for fname in os.listdir(target_dir):
        if fname.endswith('.txt'):
            continue
        fname = fname.replace('.png', '')
        fname = fname.replace('failed_', '')
        target_files.append(fname)
    target_files.sort()

    print(len(source_files), len(target_files))
    for s, t in zip(source_files, target_files):
        if s != t:
            print(s, t)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
