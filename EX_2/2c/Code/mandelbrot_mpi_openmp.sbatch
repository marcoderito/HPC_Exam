#!/bin/bash
#SBATCH --job-name="hpc_3"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=30
#SBATCH --time=02:00:00
#SBATCH --partition=THIN
#SBATCH --exclusive

# Load necessary modules
module load openMPI/4.1.5/icc/2021.7.1

# Compile the code
mpicc -o mandelbrot_mpi_openmp /orfeo/cephfs/home/dssc/mderito/osu-micro-benchmarks-7.3/mandelbrot_mpi_openmp.c -fopenmp -lm

# Run the code
srun ./mandelbrot_mpi_openmp 1920 1080 -2 -1 1 1
