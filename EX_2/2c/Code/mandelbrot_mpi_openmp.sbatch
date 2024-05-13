#!/bin/bash
#SBATCH --job-name="hpc_3"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --time=02:00:00
#SBATCH --partition=EPYC
#SBATCH --exclusive


# Load required modules
module load openmpi   # Assuming you're using Open MPI

# Set the problem size and number of threads
n_x=1920
n_y=1080
I_max=1000
threads=(1 2 4 8 16)  # List of thread configurations to test

# CSV header
echo "Threads,Problem_Size,Execution_Time" > results.csv

# Strong Scaling
for t in "${threads[@]}"; do
    echo "Running with $t threads"
    export OMP_NUM_THREADS=$t
    start_time=$(date +%s.%N)
    srun ./mandelbrot_mpi_openmp $n_x $n_y -2 -1 1 1 $I_max
    end_time=$(date +%s.%N)
    execution_time=$(echo "$end_time - $start_time" | bc)
    echo "$t,$n_x,$execution_time" >> results.csv
done

# Weak Scaling
for t in "${threads[@]}"; do
    echo "Running with $t threads and $(($n_x * $t)) x $(($n_y * $t)) problem size"
    export OMP_NUM_THREADS=$t
    start_time=$(date +%s.%N)
    srun ./mandelbrot_mpi_openmp $(($n_x * $t)) $(($n_y * $t)) -2 -1 1 1 $I_max
    end_time=$(date +%s.%N)
    execution_time=$(echo "$end_time - $start_time" | bc)
    echo "$t,$(($n_x * $t)),$execution_time" >> results.csv
done