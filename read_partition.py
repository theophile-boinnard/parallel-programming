import numpy as np
import h5py
from mpi4py import MPI
from Triangulation import Triangulation
import matplotlib.pyplot as plt
import matplotlib as mpl

N = 10

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

filename = f'mesh_{N}_rank_{rank}_out_of_{size}.h5'

with h5py.File(filename, 'r') as f:
    nodes = f['nodes'][:]
    cells = f['cells'][:]
    local_to_global = f['local_to_global'][:]
    global_to_local = f['global_to_local'][:]
    
tri = Triangulation(nodes, cells)

tri = comm.gather(tri, root=0)


if MPI.COMM_WORLD.rank==0:
    alpha = 1/size
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    cmap = 'viridis'
    norm = mpl.colors.Normalize(0, size-1)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label='rank')
    for i in range(size):
        facecolors = i*np.ones(tri[i].Nelems)
        nodecolors = i*np.ones(tri[i].Nnodes)
        ax.tripcolor(tri[i].tri_plt, facecolors=facecolors, edgecolors='black', cmap=cmap, norm=norm, alpha=alpha)
        ax.scatter(tri[i].nodes[:, 0], tri[i].nodes[:,1], c=nodecolors, cmap=cmap, norm=norm, alpha=alpha)
        
    plt.show()
