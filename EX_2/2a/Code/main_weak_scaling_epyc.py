from mpi4py import MPI
import numpy as np
from bcast_tree import binary_tree_broadcast
from bcast_flat import flat_tree_broadcast
from bcast_chain import chain_broadcast
import csv
import time

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
root_rank = 0

def measure_and_write_results(filename, results):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["num_procs", "input_size", "avg_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for num_procs, input_size, avg_time in results:
            writer.writerow({"num_procs": num_procs, "input_size": input_size, "avg_time": avg_time})

def weak_scaling():
    WARMUP_ITERATIONS = 1000
    NUM_RUNS = 1000
    num_procs = 0 
    initial_input_size = 0
    num_procs = comm.Get_size()
    input_sizes = [initial_input_size * num_procs for num_procs in range(1, num_procs + 1)]
    

    # Warmup della comunicazione
    for i in range(WARMUP_ITERATIONS):
        for input_size in input_sizes:
            binary_tree_broadcast(np.zeros(input_size), my_rank, num_procs, root_rank, 1)
            chain_broadcast(np.zeros(input_size), my_rank, num_procs, root_rank, 1)
            flat_tree_broadcast(np.zeros(input_size), my_rank, num_procs, root_rank, 1)

    # Misurazione dei tempi di esecuzione
    results = []
    for num_procs, input_size in zip(range(1, num_procs + 1), input_sizes):
        start_time = MPI.Wtime()
        for _ in range(NUM_RUNS):
            binary_tree_broadcast(np.zeros(input_size), my_rank, num_procs, root_rank, 1)
            chain_broadcast(np.zeros(input_size), my_rank, num_procs, root_rank, 1)
            flat_tree_broadcast(np.zeros(input_size), my_rank, num_procs, root_rank, 1)
        end_time = MPI.Wtime()
        avg_time = (end_time - start_time) / NUM_RUNS
        results.append((num_procs, input_size, avg_time))

    # Scrittura dei risultati su file CSV
    filename = "weak_scaling_results_epyc.csv"
    measure_and_write_results(filename, results)

if __name__ == "__main__":
    weak_scaling()

