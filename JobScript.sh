#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J swish8_400
#SBATCH --gpus-per-node=1
#SBATCH --time=23:59:0

module load anaconda3; source activate pytorch_env; export PYTHONPATH=$PYTHONPATH:/home/k/khanmu11/khanmu11/Softwares/MIST/vtk-build/lib64/python3.8/site-packages; module load gcc

python main.py

