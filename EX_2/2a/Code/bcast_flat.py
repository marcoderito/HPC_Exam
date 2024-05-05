from mpi4py import MPI
comm = MPI.COMM_WORLD
def flat_tree_broadcast(data, my_rank, num_procs, root_rank, num_elements):
    status = MPI.Status()

    if my_rank == root_rank:
        for i in range(1, num_procs):
            comm.send(data, dest=i, tag=0)
    else:
        data[:] = comm.recv(source=root_rank, tag=0, status=status)
