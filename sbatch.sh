#!/bin/bash

#SBATCH --mem=150G
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gpfs/gwg3/dpcca_v8/out.txt
#SBATCH -t 1:00:00

module load anaconda3
source activate dmcm
cd /scratch/gpfs/gwg3/dpcca_v8

python traindpcca.py --dataset=gtexv8 --latent_dim=10

