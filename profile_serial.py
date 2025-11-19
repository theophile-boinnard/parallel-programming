### profile mass matrix ###
# Théophile Boinnard, 18/11/2025 #
# Profile the serial code of mass matrix assembly #

### Import modules ###

import dolfinx 
from mpi4py import MPI
import meshio
import numpy as np

from Triangulation import Triangulation
from serial import get_mass_matrix

import time

# Square mesh #

K = np.arange(4,12)

build_mesh = []
convert_mesh = []
compute_M = []

for k in K:

    N = 2**k

    t1 = time.time()
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.triangle)
    t2 = time.time()

    nodes = mesh.geometry.x[:, :2] # Recover the nodes
    cells = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape((-1, 3)) # Recover the elems
    tri = Triangulation(nodes, cells) # Generate the triangulation
    t3 = time.time()

    M_perso = get_mass_matrix(tri)
    t4 = time.time()
    
    build_mesh.append(t2-t1)
    convert_mesh.append(t3-t2)
    compute_M.append(t4-t3)
    

N_elems = 2*(2**K)**2

for i in range(len(K)):
    total_time = build_mesh[i] + convert_mesh[i] + compute_M[i]
    print(f'Number of elements {N_elems[i]}')
    print(f'Time to build mesh     : {build_mesh[i]:3e} s | {(100*build_mesh[i]/total_time):2f}%')
    print(f'Time to convert mesh   : {convert_mesh[i]:3e} s | {(100*convert_mesh[i]/total_time):2f}%')
    print(f'Time to compute matrix : {compute_M[i]:3e} s | {(100*compute_M[i]/total_time):2f}%')
    
from scipy.stats import linregress

RES = []
for data, label in zip((build_mesh, convert_mesh, compute_M), ('Build mesh     ', 'Convert mesh   ', 'Compute matrix ')):

    res = linregress(np.log(N_elems), np.log(data))
    RES.append(res)
    print(f'{label} T = {np.exp(res.intercept)} * N^{res.slope}')
    
import matplotlib.pyplot as plt
    
fig, ax = plt.subplots()

ax.loglog(N_elems, build_mesh, 'ko', label='Build mesh')
ax.loglog(N_elems, np.exp(RES[0].intercept) * N_elems**(RES[0].slope), 'k--')
ax.loglog(N_elems, convert_mesh, 'kx', label='Convert mesh')
ax.loglog(N_elems, np.exp(RES[1].intercept) * N_elems**(RES[1].slope), 'k-.')
ax.loglog(N_elems, compute_M, 'k*', label='Compute matrix')
ax.loglog(N_elems, np.exp(RES[2].intercept) * N_elems**(RES[2].slope), 'k:')

ax.set_xlabel('Number of elements')
ax.set_ylabel('Time [s]')
ax.grid(linestyle='--')
ax.legend()
plt.tight_layout()

plt.show()
