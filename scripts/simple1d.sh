#!/bin/sh
#BSUB -q gpuv100
#BSUB -J MULT
### number of core
#BSUB -n 1
### specify that all cores should be on the same host

#BSUB -gpu "num=1:mode=exclusive_process"
### specify the memory needed
#BSUB -R "rusage[mem=10GB]"
### Number of hours needed
#BSUB -W 3:00
### added outputs and errors to files
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Error_%J.err

echo "Running script..."

module load cuda/11.8
module load python3/3.11.9
source dl/bin/activate
python3 scripts/simple1d.py > log/simpl_1d$(date +"%d-%m-%y")_$(date +'%H:%M:%S').log
