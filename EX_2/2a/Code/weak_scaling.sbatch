#!/bin/bash
#SBATCH --job-name="hpc2"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=30
#SBATCH --time=02:00:00
#SBATCH --partition=THIN
#SBATCH --exclusive


module load openMPI/4.1.5/gnu
module load openMPI/4.1.5/gnu/12.2.1
pip3 install mpi4py

mpirun -np 32 python3 /orfeo/cephfs/home/dssc/mderito/osu-micro-benchmarks-7.3/main_weak_scaling.py
