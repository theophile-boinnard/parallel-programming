### Parallel computation of mass matrix ###
# Théophile Boinnard, 10/11/2025 #
# An attempt to parallelize the computation of the mass matrix with MPI #
# The partition is done with dolfinx, which use Scotch, Kahip or Parmetis #

### Import modules ###

import dolfinx 
import ufl
import basix
from mpi4py import MPI # Necessary for dolfinx, even if we are not parallelizing yet
from dolfinx.graph import partitioner_parmetis
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix

import meshio
import numpy as np
from scipy.sparse import csr_matrix

from Triangulation import Triangulation
from serial import get_mass_matrix

import matplotlib.pyplot as plt

### Main ###

N = 4
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.triangle)
tdim = mesh.topology.dim
index_map = mesh.topology.index_map(tdim)

topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh)
nodes = geometry[:, :2]
cells = topology.reshape((-1, 4))[:, 1:]

tri = Triangulation(nodes, cells) # Generate the triangulation

I, J, V = get_mass_matrix(tri, output='data')

index_map = mesh.topology.index_map(0)
I_global = index_map.local_to_global(I)
J_global = index_map.local_to_global(J)

print(f'Rank {MPI.COMM_WORLD.rank} - Surface of sub-mesh = {np.sum(tri.elem_surf)}')
print(f'Rank {MPI.COMM_WORLD.rank} - Sum of mass matrix entries on sub-mesh = {np.sum(V)}')

I = MPI.COMM_WORLD.gather(I_global, root=0)
J = MPI.COMM_WORLD.gather(J_global, root=0)
V = MPI.COMM_WORLD.gather(V, root=0)

if MPI.COMM_WORLD.rank==0:
    
    I = np.concatenate(I)
    J = np.concatenate(J)
    V = np.concatenate(V)
    
    M = csr_matrix((V, (I, J)), shape=(np.max(I)+1, np.max(I)+1)) # Build mass matrix in csr format
    M.sum_duplicates() # Duplicate values must be summed
    
V = dolfinx.fem.functionspace(mesh, ('CG', 1)) # Continuous Galerkin P^1 space
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
m = dolfinx.fem.form(u * v * ufl.dx)
M_dolfinx = assemble_matrix(m)
M_dolfinx.assemble()

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
    
    print(f'{M.sum()=}')
    print(f'{M_dolfinx.sum()=}')
    
    #fig, axs = plt.subplots(1, 2)
    #axs[0].imshow(M.todense())
    #axs[1].imshow(M_dolfinx.todense())
    #plt.show()
    
    atol = 1e-10
    print(f'Mamimal difference between matrices entries {diff.max()}')
    assert diff.max() < atol
    print('Succesful computation of mass matrix')
    

