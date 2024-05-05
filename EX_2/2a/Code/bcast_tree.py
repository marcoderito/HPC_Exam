from mpi4py import MPI
comm = MPI.COMM_WORLD
def binary_tree_broadcast(data, my_rank, num_procs, root_rank, num_elements):
    status = MPI.Status()
    parent_rank = (my_rank - 1) // 2
    left_child_rank = 2 * my_rank + 1
    right_child_rank = 2 * my_rank + 2

    if my_rank == root_rank:
        if left_child_rank < num_procs:
            comm.send(data, dest=left_child_rank, tag=0)
        if right_child_rank < num_procs:
            comm.send(data, dest=right_child_rank, tag=0)
    else:
        data[:] = comm.recv(source=parent_rank, tag=0, status=status)
        if left_child_rank < num_procs:
            comm.send(data, dest=left_child_rank, tag=0)
        if right_child_rank < num_procs:
            comm.send(data, dest=right_child_rank, tag=0)
