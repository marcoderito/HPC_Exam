from mpi4py import MPI

def binary_tree_broadcast(data, my_rank, num_procs, root_rank, num_elements):
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    parent_rank = (my_rank - 1) // 2
    left_child_rank = 2 * my_rank + 1
    right_child_rank = 2 * my_rank + 2

    if my_rank == root_rank:
        # Invia i dati ai figli sinistro e destro, se presenti
        if left_child_rank < num_procs:
            comm.send(data, dest=left_child_rank, tag=0)
        if right_child_rank < num_procs:
            comm.send(data, dest=right_child_rank, tag=0)
    else:
        # Ricevi i dati dal genitore
        received_data = None
        if parent_rank >= 0:
            received_data = comm.recv(source=parent_rank, tag=0, status=status)
            data[:len(received_data)] = received_data
        # Invia i dati ai figli sinistro e destro, se presenti
        if left_child_rank < num_procs:
            comm.send(data, dest=left_child_rank, tag=0)
        if right_child_rank < num_procs:
            comm.send(data, dest=right_child_rank, tag=0)
