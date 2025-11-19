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
TIME = False
times = []
operations = ['Partition mesh', 'Convert to tri', 'Compute mass matrix on partition', 'Find ghosts', 'Store local values', 'Store ghost values', 'Communicate ghosts', 'Find accpeted ghosts', 'Gather all values', 'Convert to CSR', 'Convert back to coo']

def log_info(msg, out=False):
    if out:
        print(msg, flush=False)

N = 5

# Create mesh #

t1 = MPI.Wtime()
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.triangle)
times.append(MPI.Wtime() - t1)

t1 = MPI.Wtime()
tdim = mesh.topology.dim

#topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh) # Read the nodes and cells of the mesh
#nodes = geometry[:, :2] # Includes ghost nodes
#cells = topology.reshape((-1, 4))[:, 1:] # Only elems on proc

elems = mesh.topology.connectivity(tdim, 0).array.reshape((-1, 3))
nodes = mesh.geometry.x[:, :2]
elems_index_map = mesh.topology.index_map(tdim)
#cells = elems[:elems_index_map.size_local, :]
cells = dolfinx.mesh.entities_to_geometry(mesh, tdim, np.arange(elems_index_map.size_local))

tri = Triangulation(nodes, cells) # Create a triangulation for each sub-mesh
times.append(MPI.Wtime() - t1)

log_info(f'Rank {MPI.COMM_WORLD.rank} - {tri.Nnodes=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {tri.Nelems=}', DEBUG)

W = dolfinx.fem.functionspace(mesh, ('CG', 1)) # Continuous Galerkin P^1 space, needed for comparison with dolfinx

# https://fenicsproject.discourse.group/t/moving-submesh-diverges-from-its-designated-trajectory-in-a-two-phase-problem/16792
def vertex_to_dof_map_vectorized(V):
    """Create a map from the vertices of the mesh to the corresponding degree of freedom."""
    mesh = V.mesh
    num_vertices_per_cell = dolfinx.cpp.mesh.cell_num_entities(
        mesh.topology.cell_type, 0
    )

    dof_layout2 = np.empty((num_vertices_per_cell,), dtype=np.int32)
    for i in range(num_vertices_per_cell):
        var = V.dofmap.dof_layout.entity_dofs(0, i)
        assert len(var) == 1
        dof_layout2[i] = var[0]

    num_vertices = (
        mesh.topology.index_map(0).size_local + mesh.topology.index_map(0).num_ghosts
    )

    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    assert (c_to_v.offsets[1:] - c_to_v.offsets[:-1] == c_to_v.offsets[1]).all(), (
        "Single cell type supported"
    )

    vertex_to_dof_map = np.empty(num_vertices, dtype=np.int32)
    vertex_to_dof_map[c_to_v.array] = V.dofmap.list[:, dof_layout2].reshape(-1)
    return vertex_to_dof_map

vertex_to_dof_map = vertex_to_dof_map_vectorized(W)

# Compute the contribution of the cells of sub-mesh to the mass matrix

t1 = MPI.Wtime()
I, J, V = get_mass_matrix(tri, output='data') # Compute mass matrix on sub-mesh
# Some of the I, J are associated to ghost nodes! They must be send to the proper sub-matrix
mesh.topology.create_connectivity(0, 2)
times.append(MPI.Wtime() - t1)

log_info(f'Rank {MPI.COMM_WORLD.rank} - {I.shape=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {J.shape=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {I=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {J=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {V=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - Surface of sub-mesh = {np.sum(tri.elem_surf)}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - Sum of mass matrix entries on sub-mesh = {np.sum(V)}', DEBUG)

mesh.topology.create_connectivity(0, 2)
I_geom = dolfinx.mesh.entities_to_geometry(mesh, 0, I).flatten()
J_geom = dolfinx.mesh.entities_to_geometry(mesh, 0, J).flatten()
#I_geom = I.copy()
#J_geom = J.copy()

I_dofs = vertex_to_dof_map[I_geom]
J_dofs = vertex_to_dof_map[J_geom]
#I_dofs = I_geom.copy()
#J_dofs = J_geom.copy()

log_info(f'Rank {MPI.COMM_WORLD.rank} - After entites_to_geometry', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {I-I_geom=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {J-J_geom=}', DEBUG)

# Find which matrix entries belong to this rank, and which does not (ghosts) #

t1 = MPI.Wtime()
index_map = mesh.topology.index_map(0)
local_indices = np.where(I_geom<index_map.size_local) # On current rank, we only keep the rows of the owned nodes
ghost_indices = np.where(I_geom>=index_map.size_local) # The rows of ghost nodes must be comunicated to other ranks
times.append(MPI.Wtime() - t1)

log_info(f'Rank {MPI.COMM_WORLD.rank} - {dolfinx.mesh.entities_to_geometry(mesh, 0, np.arange(tri.Nnodes)).flatten()=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {local_indices=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {ghost_indices=}', DEBUG)

# Keep these value on this rank #

t1 = MPI.Wtime()
I_local = I_dofs[local_indices] 
J_local = J_dofs[local_indices]
V_local = V[local_indices]

log_info(f'Rank {MPI.COMM_WORLD.rank} - {I_local=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {J_local=}', DEBUG)

# column index is global to each rank, so we convert it straight away #

J_global = index_map.local_to_global(J_local)
times.append(MPI.Wtime() - t1)

# Extract values associated to ghost nodes #

t1 = MPI.Wtime()
I_ghost = I_dofs[ghost_indices]
J_ghost = J_dofs[ghost_indices]
V_ghost = V[ghost_indices]

# Convert local indexing into global indexing before communication #

I_ghost_global = index_map.local_to_global(I_ghost)
J_ghost_global = index_map.local_to_global(J_ghost)
times.append(MPI.Wtime() - t1)

# Send ghost nodes to all other ranks #

log_info(f'Rank {MPI.COMM_WORLD.rank} - Before allgather - {I_ghost_global=}', DEBUG)

t1 = MPI.Wtime()
I_ghost_global = np.concatenate(MPI.COMM_WORLD.allgather(I_ghost_global)) # we do not know to which rank send the ghost nodes, so we send it to everyone
J_ghost_global = np.concatenate(MPI.COMM_WORLD.allgather(J_ghost_global))
V_ghost = np.concatenate(MPI.COMM_WORLD.allgather(V_ghost))
times.append(MPI.Wtime() - t1)

log_info(f'Rank {MPI.COMM_WORLD.rank} - After allgather - {I_ghost_global=}', DEBUG)

# The current rank have recieved values, it has to check if they are intended to him #

t1 = MPI.Wtime()
global_indices = index_map.local_to_global(np.arange(index_map.size_local)) # If a global index of recieved ghost point is not in this array, it must be removed

log_info(f'Rank {MPI.COMM_WORLD.rank} - {global_indices=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {np.arange(index_map.size_local) + index_map.local_range[0]=}', DEBUG)

accepted_vertices = np.isin(I_ghost_global, global_indices)
I_ghost_global_accepted = I_ghost_global[accepted_vertices]
J_ghost_global_accepted = J_ghost_global[accepted_vertices]
V_ghost_accepted = V_ghost[accepted_vertices]
times.append(MPI.Wtime() - t1)

log_info(f'Rank {MPI.COMM_WORLD.rank} - Ghost kept - {I_ghost_global_accepted=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - Ghost kept - {J_ghost_global_accepted=}', DEBUG)

# Convert back local global to local #
# Only row index I is local, colum index J is global #

t1 = MPI.Wtime()
I_ghost_local_accepted = index_map.global_to_local(I_ghost_global_accepted)
times.append(MPI.Wtime() - t1)

log_info(f'Rank {MPI.COMM_WORLD.rank} - {I_ghost_local_accepted=}', DEBUG)     

# Concatenate computed values and recieved values #

t1 = MPI.Wtime()
I_final = np.concatenate((I_local, I_ghost_local_accepted))
J_final = np.concatenate((J_global, J_ghost_global_accepted))
V_final = np.concatenate((V_local, V_ghost_accepted))
times.append(MPI.Wtime() - t1)

log_info(f'Rank {MPI.COMM_WORLD.rank} - {I_final=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {J_final=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {V_final=}', DEBUG)

log_info(f'Rank {MPI.COMM_WORLD.rank} - {I_final.min()=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {I_final.max()=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {J_final.min()=}', DEBUG)
log_info(f'Rank {MPI.COMM_WORLD.rank} - {J_final.max()=}', DEBUG)

# Create CSR matrix to have automatic summation of repeated pair of indices #

t1 = MPI.Wtime()
M = csr_matrix((V_final, (I_final, J_final)), shape=(index_map.size_local, index_map.size_global)) # automatic sum of duplicate entries
# M.sum_duplicates()
times.append(MPI.Wtime() - t1)

# Convert back to coordinates format #

t1 = MPI.Wtime()
M = M.tocoo()
I_sum_duplicates = M.row
J_sum_duplicates = M.col
V_sum_duplicates = M.data
times.append(MPI.Wtime() - t1)
    
# Verify with dolfinx #

if not TIME:
    
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
    
    log_info(f'Rank {MPI.COMM_WORLD.rank} - {vertex_to_dof_map=}', DEBUG)
    #I_reorder = vertex_to_dof_map[I_sum_duplicates]
    #J_reorder = vertex_to_dof_map[J_sum_duplicates]
    I_reorder = I_sum_duplicates.copy()
    J_reorder = J_sum_duplicates.copy()
    M = csr_matrix((V_sum_duplicates, (I_reorder, J_reorder)), shape=(index_map.size_local, (N+1)**2))

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

    I_global_final = index_map.local_to_global(I_sum_duplicates)
    J_global_final = J_sum_duplicates

    I_all = MPI.COMM_WORLD.gather(I_global_final, root=0)
    J_all = MPI.COMM_WORLD.gather(J_global_final, root=0)
    V_all = MPI.COMM_WORLD.gather(V_sum_duplicates, root=0)

    # Show full mass matrix #

    if MPI.COMM_WORLD.rank==0:
        
        I0 = np.concatenate(I_all)
        J0 = np.concatenate(J_all)
        V = np.concatenate(V_all)
        
        M = csr_matrix((V, (I0, J0)), shape=(index_map.size_global, index_map.size_global)) # Build mass matrix in csr format
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
        
if TIME:
    print(f'On rank {MPI.COMM_WORLD.rank}')
    max_op_len = max(len(op) for op in operations)
    for operation, time in zip(operations, times):
        print(f'{operation:<{max_op_len}} : {time} s')
    
    total_time = np.sum(times)
    print(f'Condensed')
    print(f'Partition      : {times[0]:3e} s | {(100*times[0]/total_time):2f}%')
    print(f'Compute matrix : {np.sum(times[1:]):3e} s | {(100*np.sum(times[1:])/total_time):2f}%')
        

