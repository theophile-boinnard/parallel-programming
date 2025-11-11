### Verify mass matrix ###
# Théophile Boinnard, 10/11/2025 #
# A code to verify my implementation of the mass matrix in serial, against dolfinx #

### Import modules ###

import dolfinx 
import ufl
import basix
from mpi4py import MPI # Necessary for dolfinx, even if we are not parallelizing yet

import meshio
import numpy as np
from scipy.sparse import csr_matrix

from Triangulation import Triangulation
from serial import get_mass_matrix

### Compute mass matrix with dolfinx ###

def get_mass_matrix_dolfinx(mesh):
    '''
    Compute the mass matrix using dolfinx
    Input : 
    - mesh : a dolfinx mesh
    Output : 
    - M : mass matrix
    '''
    
    V = dolfinx.fem.functionspace(mesh, ('CG', 1)) # Continuous Galerkin P^1 space
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    m = dolfinx.fem.form(u * v * ufl.dx)
    M = dolfinx.fem.assemble_matrix(m)
    # M.assemble()
    
    return M
    
### Compare results ###

# Reading a mesh #

filename = 'disk_1.mesh'
read_mesh = meshio.read(filename)

c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
nodes = read_mesh.points[:, :2]
elems = read_mesh.cells_dict['triangle']
mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells=elems, x=nodes, e=c_el) # dolfinx shuffle nodes and elements
nodes = mesh.geometry.x[:, :2] # Recover the nodes
cells = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape((-1, 3)) # Recover the elems
tri = Triangulation(nodes, cells) # Generate the triangulation

M_perso = get_mass_matrix(tri)
M_dolfinx = get_mass_matrix_dolfinx(mesh)

assert np.allclose(M_perso.toarray(), M_dolfinx.to_dense())

print('The serial implementation of the mass matrix is correct.')
