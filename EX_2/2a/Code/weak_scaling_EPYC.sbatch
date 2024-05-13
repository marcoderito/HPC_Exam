#!/bin/bash
#SBATCH --job-name="hpc11"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --time=02:00:00
#SBATCH --partition=EPYC
#SBATCH --exclusive


module load openMPI/4.1.5/gnu
module load openMPI/4.1.5/gnu/12.2.1
pip3 install mpi4py

mpirun -np 32 python3 /orfeo/cephfs/home/dssc/mderito/osu-micro-benchmarks-7.3/main_weak_scaling_epyc.py
