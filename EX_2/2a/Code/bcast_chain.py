from mpi4py import MPI
comm = MPI.COMM_WORLD
def chain_broadcast(data, my_rank, num_procs, root_rank, num_elements):
    status = MPI.Status()
    parent_rank = my_rank - 1
    child_rank = my_rank + 1

    if my_rank == root_rank:
        comm.send(data, dest=child_rank, tag=0)
    else:
        data[:] = comm.recv(source=parent_rank, tag=0, status=status)
        if child_rank < num_procs:
            comm.send(data, dest=child_rank, tag=0)
