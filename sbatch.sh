#!/bin/bash

#SBATCH --mem 250G
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:2
#SBATCH -o /scratch/gpfs/gwg3/dmcm/out.txt
#SBATCH -t 1:00:00
#SBATCH --mail-user=ggundersen@princeton.edu

module load anaconda3
source activate dmcm

cd /scratch/gpfs/gwg3/dmcm

python trainpcca.py --dataset=mnist --latent_dim=2

