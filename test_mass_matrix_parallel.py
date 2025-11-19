### Timing of parallel computation of mass matrix ###
# Théophile Boinnard, 18/11/2025 #

### Import modules ###

import dolfinx 
import ufl
import basix
from mpi4py import MPI
from dolfinx.graph import partitioner_parmetis
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix

import meshio
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from Triangulation import Triangulation
from serial import get_mass_matrix

import matplotlib.pyplot as plt

### Main ###

N = int(4096*np.sqrt(2))

# Create mesh #

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.triangle)

MPI.COMM_WORLD.Barrier()
t1 = MPI.Wtime()

tdim = mesh.topology.dim

topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh) # Read the nodes and cells of the mesh
nodes = geometry[:, :2] # Includes ghost nodes
cells = topology.reshape((-1, 4))[:, 1:] # Only elems on proc

#elems = mesh.topology.connectivity(tdim, 0).array.reshape((-1, 3))
#nodes = mesh.geometry.x[:, :2]
#elems_index_map = mesh.topology.index_map(tdim)
#cells = elems[:elems_index_map.size_local, :]

tri = Triangulation(nodes, cells) # Create a triangulation for each sub-mesh

# Compute the contribution of the cells of sub-mesh to the mass matrix

I, J, V = get_mass_matrix(tri, output='data') # Compute mass matrix on sub-mesh
# Some of the I, J are associated to ghost nodes! They must be send to the proper sub-matrix

# Find which matrix entries belong to this rank, and which does not (ghosts) #

index_map = mesh.topology.index_map(0)
local_indices = np.where(I<index_map.size_local) # On current rank, we only keep the rows of the owned nodes
ghost_indices = np.where(I>=index_map.size_local) # The rows of ghost nodes must be comunicated to other ranks

# Keep these value on this rank #

I_local = I[local_indices] 
J_local = J[local_indices]
V_local = V[local_indices]

# column index is global to each rank, so we convert it straight away #

J_global = index_map.local_to_global(J_local)

# Extract values associated to ghost nodes #

I_ghost = I[ghost_indices]
J_ghost = J[ghost_indices]
V_ghost = V[ghost_indices]

# Convert local indexing into global indexing before communication #

I_ghost_global = index_map.local_to_global(I_ghost)
J_ghost_global = index_map.local_to_global(J_ghost)

# Send ghost nodes to all other ranks #

I_ghost_global = np.concatenate(MPI.COMM_WORLD.allgather(I_ghost_global)) # we do not know to which rank send the ghost nodes, so we send it to everyone
J_ghost_global = np.concatenate(MPI.COMM_WORLD.allgather(J_ghost_global))
V_ghost = np.concatenate(MPI.COMM_WORLD.allgather(V_ghost))

# The current rank have recieved values, it has to check if they are intended to him #

global_indices = index_map.local_to_global(np.arange(index_map.size_local)) # If a global index of recieved ghost point is not in this array, it must be removed

accepted_vertices = np.isin(I_ghost_global, global_indices)
I_ghost_global_accepted = I_ghost_global[accepted_vertices]
J_ghost_global_accepted = J_ghost_global[accepted_vertices]
V_ghost_accepted = V_ghost[accepted_vertices]

# Convert back local global to local #
# Only row index I is local, colum index J is global #

I_ghost_local_accepted = index_map.global_to_local(I_ghost_global_accepted)

# Concatenate computed values and recieved values #

I_final = np.concatenate((I_local, I_ghost_local_accepted))
J_final = np.concatenate((J_global, J_ghost_global_accepted))
V_final = np.concatenate((V_local, V_ghost_accepted))

# Create CSR matrix to have automatic summation of repeated pair of indices #

M = csr_matrix((V_final, (I_final, J_final)), shape=(index_map.size_local, index_map.size_global)) # automatic sum of duplicate entries
# M.sum_duplicates()

# Convert back to coordinates format #

M = M.tocoo()
I = M.row
J = M.col
V = M.data

MPI.COMM_WORLD.Barrier()
t2 = MPI.Wtime()

if MPI.COMM_WORLD.rank == 0:
    print("Mass matrix assembly time:", t2 - t1)
