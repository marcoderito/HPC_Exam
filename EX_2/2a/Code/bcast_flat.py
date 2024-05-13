from mpi4py import MPI

def flat_tree_broadcast(data, my_rank, num_procs, root_rank, num_elements):
    comm = MPI.COMM_WORLD
    status = MPI.Status()

    if my_rank == root_rank:
        # Invia i dati a tutti i processi tranne alla radice
        for i in range(num_procs):
            if i != root_rank:
                comm.send(data, dest=i, tag=0)
    else:
        # Ricevi i dati dalla radice
        received_data = comm.recv(source=root_rank, tag=0, status=status)
        data[:len(received_data)] = received_data
