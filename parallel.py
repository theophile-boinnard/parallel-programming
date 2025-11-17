### Parallel computation of mass matrix ###
# Théophile Boinnard, 10/11/2025 #
# An attempt to parallelize the computation of the mass matrix with MPI #
# The partition is done with dolfinx, which use Scotch, Kahip or Parmetis #

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

DEBUG = True
PLOT = True

def log_info(msg, out=False):
    if out:
        print(msg, flush=False)

N = 5

# Create mesh #

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.triangle)
tdim = mesh.topology.dim

topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh) # Read the nodes and cells of the mesh
nodes = geometry[:, :2] # Includes ghost nodes
cells = topology.reshape((-1, 4))[:, 1:] # Only elems on proc

tri = Triangulation(nodes, cells) # Create a triangulation for each sub-mesh

log_info(f'Rank {MPI.COMM_WORLD.rank} - {tri.Nnodes=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {tri.Nelems=}', DEBUG)

# Compute the contribution of the cells of sub-mesh to the mass matrix

I, J, V = get_mass_matrix(tri, output='data') # Compute mass matrix on sub-mesh
# Some of the I, J are associated to ghost nodes! They must be send to the proper sub-matrix

log_info(f'Rank {MPI.COMM_WORLD.rank} - {I.shape=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {J.shape=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {I=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {J=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {V=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - Surface of sub-mesh = {np.sum(tri.elem_surf)}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - Sum of mass matrix entries on sub-mesh = {np.sum(V)}', DEBUG)

# Find which matrix entries belong to this rank, and which does not (ghosts) #

index_map = mesh.topology.index_map(0)
local_indices = np.where(I<index_map.size_local) # On current rank, we only keep the rows of the owned nodes
ghost_indices = np.where(I>=index_map.size_local) # The rows of ghost nodes must be comunicated to other ranks

log_info(f'Rank {MPI.COMM_WORLD.rank} - {local_indices=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {ghost_indices=}', DEBUG)

# Keep these value on this rank #

I_local = I[local_indices] 
J_local = J[local_indices]
V_local = V[local_indices]

log_info(f'Rank {MPI.COMM_WORLD.rank} - {I_local=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {J_local=}', DEBUG)

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

log_info(f'Rank {MPI.COMM_WORLD.rank} - Before allgather - {I_ghost_global=}', DEBUG)

I_ghost_global = np.concatenate(MPI.COMM_WORLD.allgather(I_ghost_global)) # we do not know to which rank send the ghost nodes, so we send it to everyone
J_ghost_global = np.concatenate(MPI.COMM_WORLD.allgather(J_ghost_global))
V_ghost = np.concatenate(MPI.COMM_WORLD.allgather(V_ghost))

log_info(f'Rank {MPI.COMM_WORLD.rank} - After allgather - {I_ghost_global=}', DEBUG)

# The current rank have recieved values, it has to check if they are intended to him #

global_indices = index_map.local_to_global(np.arange(index_map.size_local)) # If a global index of recieved ghost point is not in this array, it must be removed

log_info(f'Rank {MPI.COMM_WORLD.rank} - {global_indices=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {np.arange(index_map.size_local) + index_map.local_range[0]=}', DEBUG)

accepted_vertices = np.isin(I_ghost_global, global_indices)
I_ghost_global_accepted = I_ghost_global[accepted_vertices]
J_ghost_global_accepted = J_ghost_global[accepted_vertices]
V_ghost_accepted = V_ghost[accepted_vertices]

log_info(f'Rank {MPI.COMM_WORLD.rank} - Ghost kept - {I_ghost_global_accepted=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - Ghost kept - {J_ghost_global_accepted=}', DEBUG)

# Convert back local global to local #
# Only row index I is local, colum index J is global #

I_ghost_local_accepted = index_map.global_to_local(I_ghost_global_accepted)

log_info(f'Rank {MPI.COMM_WORLD.rank} - {I_ghost_local_accepted=}', DEBUG)     

# Concatenate computed values and recieved values #

I_final = np.concatenate((I_local, I_ghost_local_accepted))
J_final = np.concatenate((J_global, J_ghost_global_accepted))
V_final = np.concatenate((V_local, V_ghost_accepted))

log_info(f'Rank {MPI.COMM_WORLD.rank} - {I_final=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {J_final=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {V_final=}', DEBUG)

log_info(f'Rank {MPI.COMM_WORLD.rank} - {I_final.min()=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {I_final.max()=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {J_final.min()=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {J_final.max()=}', DEBUG)

# Create CSR matrix to have automatic summation of repeated pair of indices #

M = csr_matrix((V_final, (I_final, J_final)), shape=(index_map.size_local, index_map.size_global)) # automatic sum of duplicate entries
# M.sum_duplicates()

# Convert back to coordinates format #

M = M.tocoo()
I = M.row
J = M.col
V = M.data
    
# Verify with dolfinx #
    
W = dolfinx.fem.functionspace(mesh, ('CG', 1)) # Continuous Galerkin P^1 space
u = ufl.TrialFunction(W)
v = ufl.TestFunction(W)
m = dolfinx.fem.form(u * v * ufl.dx)
M_dolfinx = assemble_matrix(m)
M_dolfinx.assemble()

# Show sub matrices #

# https://fenicsproject.discourse.group/t/conversion-from-array-to-petsc-and-from-petsc-to-array/6941
def petsc2array(v):
    s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
    return s

def local_petsc_matrix(A: PETSc.Mat):
    # Row ownership range
    r0, r1 = A.getOwnershipRange()

    # Full column range
    ncols = A.getSize()[1]
    cols = range(ncols)

    # Extract values (this returns a NumPy array)
    local_block = A.getValues(range(r0, r1), cols)
    return local_block

log_info(f'Rank {MPI.COMM_WORLD.rank} - {mesh.geometry.input_global_indices.max()=}', DEBUG)
Md = local_petsc_matrix(M_dolfinx)
#M = csr_matrix((V, (I, J)), shape=(index_map.size_local, (N+1)**2))

diff = np.abs(Md - M.todense())
atol = 1e-10

if PLOT:

    log_info(f'Rank {MPI.COMM_WORLD.rank} - {np.sum(M.todense())=}', DEBUG) 
    log_info(f'Rank {MPI.COMM_WORLD.rank} - {np.sum(Md)=}', DEBUG) 
    
    fig, axs = plt.subplots(3, 1)
    axs[0].imshow(np.heaviside(M.todense(), 0))
    axs[0].set_title('M perso')
    axs[1].imshow(np.heaviside(Md, 0))
    axs[1].set_title('M dolfinx')
    axs[2].imshow(diff>atol)
    axs[2].set_title('difference')
    fig.suptitle(f'Rank {MPI.COMM_WORLD.rank}')
    plt.tight_layout()
    plt.show()

# Global indices to reconstruct full matrix #

I_global = index_map.local_to_global(I)
J_global = J

I_all = MPI.COMM_WORLD.gather(I_global, root=0)
J_all = MPI.COMM_WORLD.gather(J_global, root=0)
V_all = MPI.COMM_WORLD.gather(V, root=0)

# Show full mass matrix #

if MPI.COMM_WORLD.rank==0:
    
    I = np.concatenate(I_all)
    J = np.concatenate(J_all)
    V = np.concatenate(V_all)
    
    M = csr_matrix((V, (I, J)), shape=(index_map.size_global, index_map.size_global)) # Build mass matrix in csr format
    # M.sum_duplicates() # Duplicate values must be summed

# Taken from https://github.com/jorgensd/dolfinx_mpc/blob/main/python/src/dolfinx_mpc/utils/test.py
# Again, thanks dokken
def gather_PETScMatrix(A: PETSc.Mat, root=0) -> csr_matrix:  # type: ignore
    """
    Given a distributed PETSc matrix, gather in on process 'root' in
    a scipy CSR matrix
    """
    ai, aj, av = A.getValuesCSR()
    aj_all = MPI.COMM_WORLD.gather(aj, root=root)  # type: ignore
    av_all = MPI.COMM_WORLD.gather(av, root=root)  # type: ignore
    ai_all = MPI.COMM_WORLD.gather(ai, root=root)  # type: ignore
    if MPI.COMM_WORLD.rank == root:
        ai_cum = [0]
        for ai in ai_all:  # type: ignore
            offsets = ai[1:] + ai_cum[-1]
            ai_cum.extend(offsets)
        return csr_matrix((np.hstack(av_all), np.hstack(aj_all), ai_cum), shape=A.getSize())  # type: ignore
        
M_dolfinx = gather_PETScMatrix(M_dolfinx, root=0)
        
if MPI.COMM_WORLD.rank==0:
    diff = np.abs(M_dolfinx - M)
    atol = 1e-10
    
    print(f'{M.nnz=}')
    print(f'{M_dolfinx.nnz=}')
    
    print(f'{M.sum()=}')
    print(f'{M_dolfinx.sum()=}')
    
    if PLOT:
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(np.heaviside(M.todense(), 0))
        axs[0].set_title('M perso')
        axs[1].imshow(np.heaviside(M_dolfinx.todense(), 0))
        axs[1].set_title('M dolfinx')
        axs[2].imshow(diff.todense()>atol)
        axs[2].set_title('difference')
        plt.show()
    
    print(f'Mamimal difference between matrices entries {diff.max()}')
    if diff.max() >= atol:
        I, J = np.where(diff.todense() >= atol)
        print(f'Indices at fault :')
        print(f'{I=}')
        print(f'{J=}')
    else:
        print('Succesful computation of mass matrix')

