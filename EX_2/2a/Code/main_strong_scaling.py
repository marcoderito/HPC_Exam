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
        fieldnames = ["num_procs", "avg_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for num_procs, avg_time in results.items():
            writer.writerow({"num_procs": num_procs, "avg_time": avg_time})

def strong_scaling():
    WARMUP_ITERATIONS = 1000
    NUM_RUNS = 1000
    initial_data_size = 10000  # Dimensione iniziale dei dati di esempio

    # Ottieni il numero totale di processi
    num_procs = comm.Get_size()

    # Warmup della comunicazione
    for i in range(WARMUP_ITERATIONS):
        binary_tree_broadcast(np.zeros(initial_data_size), my_rank, num_procs, root_rank, 1)
        chain_broadcast(np.zeros(initial_data_size), my_rank, num_procs, root_rank, 1)
        flat_tree_broadcast(np.zeros(initial_data_size), my_rank, num_procs, root_rank, 1)

    results = {}
    for num_procs in range(1, num_procs + 1):
        input_size = initial_data_size // num_procs  # Regola la dimensione dei dati per processo
        avg_time = 0
        for _ in range(NUM_RUNS):
            start_time = time.time()  # Misura il tempo di inizio all'interno del ciclo
            # Esegue il benchmark utilizzando la funzione di broadcast specifica
            # (binary_tree_broadcast, chain_broadcast, flat_tree_broadcast)
            end_time = time.time()
            avg_time += (end_time - start_time) / NUM_RUNS
        results[num_procs] = avg_time


    # Scrittura dei risultati su file CSV
    filename = "strong_scaling_results.csv"
    measure_and_write_results(filename, results)


if __name__ == "__main__":
    strong_scaling()
