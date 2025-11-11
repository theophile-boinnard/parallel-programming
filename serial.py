### Mass matrix assembly in serial ###
# Théophile Boinnard, 10/11/2025 #

### Import modules ###

from Triangulation import Triangulation
import meshio
from scipy.sparse import csr_matrix
import numpy as np

### Main ###

def get_mass_matrix(tri : Triangulation, output : str = 'matrix'):
    '''
    Compute mass matrix in serial, using Python vectorization
    Input : 
    - tri : a triangulation
    - output : if "matrix", returns M as scipy.sparse.scr_matrix, if "data" give the indices and values
    Output : 
    - M [if output=="matrix"] : mass matrix as scipy.sparse.scr_matrix
    - I, J, V [if output=="data"] : indices and values to build mass matrix
    '''
    
    assert output in ['matrix', 'data']

    M_ref = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 24 # P^1 mass matrix on reference element
    BK, _ = tri.linear_maps # Linear mapping from reference element to elements
    detB = np.abs(np.linalg.det(BK)) # Absolute value of determinant of the map

    M_loc_all = detB[:, None, None] * M_ref[None, :, :] # Mass matrix contribution on each element

    I = np.repeat(tri.elems, 3, axis=1) # i indices
    J = np.tile(tri.elems, (1, 3)) # j indices
    V = M_loc_all.reshape(-1, 9) # values corresponding to i, j

    if output=='matrix':
        M = csr_matrix((V.ravel(), (I.ravel(), J.ravel())), shape=(tri.Nnodes, tri.Nnodes)) # Build mass matrix in csr format
        M.sum_duplicates() # Duplicate values must be summed
        return M
        
    if output=='data':
        return I.ravel(), J.ravel(), V.ravel()
