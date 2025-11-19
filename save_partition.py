### Save partition to run on cluster ###
# Théophile Boinnard, 19/11/2025 #

### Import modules ###

import dolfinx 
import ufl
import basix
from mpi4py import MPI
import numpy as np

import h5py

### Main ###

N = 8192
#N = int(4096*np.sqrt(2))

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.triangle)
tdim = mesh.topology.dim

elems = mesh.topology.connectivity(tdim, 0).array.reshape((-1, 3))
nodes = mesh.geometry.x[:, :2]
elems_index_map = mesh.topology.index_map(tdim)
cells = dolfinx.mesh.entities_to_geometry(mesh, tdim, np.arange(elems_index_map.size_local))

index_map = mesh.topology.index_map(0)
local_to_global = index_map.local_to_global(np.arange(index_map.size_local + index_map.num_ghosts))
global_to_local = index_map.global_to_local(np.arange(index_map.size_global))

filename = f'mesh_{N}/mesh_{N}_rank_{rank}_out_of_{size}.h5'

with h5py.File(filename, 'w') as f:
    f.create_dataset('nodes', data=nodes, dtype='float64')
    f.create_dataset('cells', data=cells, dtype='int')
    f.create_dataset('local_to_global', data=local_to_global, dtype='int')
    f.create_dataset('global_to_local', data=global_to_local, dtype='int')
    f.create_dataset('size_local', data=index_map.size_local, dtype='int')
    
print(f'Succesfully save {filename}')
