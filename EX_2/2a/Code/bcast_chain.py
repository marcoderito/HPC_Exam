from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD


def chain_broadcast(data, my_rank, num_procs, root_rank, iteration):
    """Esegue la trasmissione a catena con una dimensione dei dati coerente sul processo root.

    Args:
        data: I dati da trasmettere (array sul processo root, vuoto sugli altri).
        my_rank: Il rank del processo corrente.
        num_procs: Il numero totale di processi.
        root_rank: Il rank del processo root.
        iteration: Contatore di iterazione (opzionale, non utilizzato in questa implementazione).
    """

    status = MPI.Status()
    parent_rank = my_rank - 1
    child_rank = my_rank + 1

    if my_rank == root_rank:
        # Usa la dimensione iniziale dei dati indipendentemente dal numero di processi (Opzione 1)
        data = np.zeros(initial_data_size)
        comm.send(data, dest=child_rank, tag=0)
    else:
        received_data = comm.recv(source=parent_rank, tag=0, status=status)
        if received_data.shape[0] != initial_data_size:
            print("Error: Received data shape does not match expected size.")
            # Gestione dell'errore: interrompere il programma o gestire l'errore in altro modo
        else:
            data[:] = received_data
            if child_rank < num_procs:
                comm.send(data, dest=child_rank, tag=0)
