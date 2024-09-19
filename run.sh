#!/bin/bash
#SBATCH -J tranBSub
#SBATCH --output=bin_muse.txt
#SBATCH -p p-A100
#SBATCH -A t00120220002
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1


# source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh 
# conda config --add envs_dirs /mntnfs/lee_data1/lijunjie/anaconda3/envs
# conda activate py1.11
nvidia-smi


python repeat_exp_main3.py
# python repeat_exp_main4.py 
# python repeat_exp_main2.py & 
# python repeat_exp_main4.py &
# python repeat_exp_main6.py &
# python repeat_exp_main8.py &
# python repeat_exp_main.py &
# python repeat_exp_main3.py &
# python repeat_exp_main5.py &
# python repeat_exp_main7.py

