### Parallel computation of mass matrix ###
# Théophile Boinnard, 19/11/2025 #
# Reads a partition savefd on an h5 file #

### Import modules ###

from mpi4py import MPI
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from Triangulation import Triangulation
from serial import get_mass_matrix
import matplotlib.pyplot as plt
import h5py

### Main ###

TIME = True
times = []
operations = ['Read mesh', 'Compute mass matrix on partition', 'Find ghosts', 'Store local values', 'Store ghost values', 'Communicate ghosts', 'Find accpeted ghosts', 'Convert I to local', 'Concatenate all values', 'Convert to CSR', 'Convert back to coo']

N = 1024
#N = int(4096*np.sqrt(2))

# Create mesh #

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

comm.Barrier()
T1 = MPI.Wtime()

t1 = MPI.Wtime()
filename = f'/scratch/boinnard/phys-743/mesh_{N}/mesh_{N}_rank_{rank}_out_of_{size}.h5'

with h5py.File(filename, 'r') as f:
    nodes = f['nodes'][:]
    cells = f['cells'][:]
    local_to_global = f['local_to_global'][:]
    global_to_local = f['global_to_local'][:]
    size_local = f['size_local'][()]
    
tri = Triangulation(nodes, cells)

times.append(MPI.Wtime() - t1)

# Compute the contribution of the cells of sub-mesh to the mass matrix

t1 = MPI.Wtime()
I, J, V = get_mass_matrix(tri, output='data') # Compute mass matrix on sub-mesh
# Some of the I, J are associated to ghost nodes! They must be send to the proper sub-matrix
times.append(MPI.Wtime() - t1)

# Find which matrix entries belong to this rank, and which does not (ghosts) #

t1 = MPI.Wtime()
local_indices = np.where(I<size_local) # On current rank, we only keep the rows of the owned nodes
ghost_indices = np.where(I>=size_local) # The rows of ghost nodes must be comunicated to other ranks
times.append(MPI.Wtime() - t1)

# Keep these value on this rank #

t1 = MPI.Wtime()
I_local = I[local_indices] 
J_local = J[local_indices]
V_local = V[local_indices]

# column index is global to each rank, so we convert it straight away #

J_global = local_to_global[J_local]
times.append(MPI.Wtime() - t1)

# Extract values associated to ghost nodes #

t1 = MPI.Wtime()
I_ghost = I[ghost_indices]
J_ghost = J[ghost_indices]
V_ghost = V[ghost_indices]

# Convert local indexing into global indexing before communication #

I_ghost_global = local_to_global[I_ghost]
J_ghost_global = local_to_global[J_ghost]
times.append(MPI.Wtime() - t1)

# Send ghost nodes to all other ranks #

t1 = MPI.Wtime()
I_ghost_global = np.concatenate(comm.allgather(I_ghost_global)) # we do not know to which rank send the ghost nodes, so we send it to everyone
J_ghost_global = np.concatenate(comm.allgather(J_ghost_global))
V_ghost = np.concatenate(comm.allgather(V_ghost))
times.append(MPI.Wtime() - t1)

# The current rank have recieved values, it has to check if they are intended to him #

t1 = MPI.Wtime()
global_indices = local_to_global[np.arange(size_local)] # If a global index of recieved ghost point is not in this array, it must be removed

accepted_vertices = np.isin(I_ghost_global, global_indices)
I_ghost_global_accepted = I_ghost_global[accepted_vertices]
J_ghost_global_accepted = J_ghost_global[accepted_vertices]
V_ghost_accepted = V_ghost[accepted_vertices]
times.append(MPI.Wtime() - t1)

# Convert back local global to local #
# Only row index I is local, colum index J is global #

t1 = MPI.Wtime()
I_ghost_local_accepted = global_to_local[I_ghost_global_accepted]
times.append(MPI.Wtime() - t1)   

# Concatenate computed values and recieved values #

t1 = MPI.Wtime()
I_final = np.concatenate((I_local, I_ghost_local_accepted))
J_final = np.concatenate((J_global, J_ghost_global_accepted))
V_final = np.concatenate((V_local, V_ghost_accepted))
times.append(MPI.Wtime() - t1)

# Create CSR matrix to have automatic summation of repeated pair of indices #

t1 = MPI.Wtime()
M = csr_matrix((V_final, (I_final, J_final)), shape=(size_local, global_to_local.shape[0])) # automatic sum of duplicate entries
# M.sum_duplicates()
times.append(MPI.Wtime() - t1)

# Convert back to coordinates format #

t1 = MPI.Wtime()
M = M.tocoo()
I_sum_duplicates = M.row
J_sum_duplicates = M.col
V_sum_duplicates = M.data
times.append(MPI.Wtime() - t1)
    
if TIME:
    print(f'On rank {MPI.COMM_WORLD.rank}')
    max_op_len = max(len(op) for op in operations)
    for operation, time in zip(operations, times):
        print(f'{operation:<{max_op_len}} : {time} s')
    
    total_time = np.sum(times)
    print(f'Condensed')
    print(f'Read mesh      : {times[0]:3e} s | {(100*times[0]/total_time):2f}%')
    print(f'Compute matrix : {times[1]:3e} s | {(100*times[1]/total_time):2f}%')
    print(f'Communication  : {np.sum(times[2:]):3e} s | {(100*np.sum(times[2:])/total_time):2f}%')
        
comm.Barrier()
T2 = MPI.Wtime()

ttimes = np.array(comm.gather(times, root=0))

if rank==0:
    times = np.mean(ttimes, axis=0)
    total_time = np.sum(times)
    print(f'Total execution time {T2-T1} s')
    print(f'Read mesh      : {times[0]:3e} s | {(100*times[0]/total_time):2f}%')
    print(f'Compute matrix : {times[1]:3e} s | {(100*times[1]/total_time):2f}%')
    print(f'Communication  : {np.sum(times[2:]):3e} s | {(100*np.sum(times[2:])/total_time):2f}%')
