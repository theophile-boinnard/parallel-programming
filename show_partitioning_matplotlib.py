### Show mesh partitioning ###
# Théophile Boinnard, 10/11/2025 #
# Show the mesh partition using matplotlib #
# The partition is done with dolfinx, which use Scotch, Kahip or Parmetis #

### Import modules ###

import dolfinx 
import ufl
import basix
from mpi4py import MPI # Necessary for dolfinx, even if we are not parallelizing yet
from dolfinx.graph import partitioner_parmetis
partitioner = dolfinx.mesh.create_cell_partitioner(partitioner_parmetis())
from dolfinx.io import XDMFFile, gmsh as gmshio

import meshio
import numpy as np
from scipy.sparse import csr_matrix

from Triangulation import Triangulation

import matplotlib.pyplot as plt
import matplotlib as mpl

### Main ###

mesh_type = 'from_file' # either 'from_file' or 'square'

assert mesh_type in ['from_file', 'square']

if mesh_type=='from_file':

    name = 'disk_1'
    extension = '.mesh'
    filename = name + extension

    # From https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html
    # Thanks dokken

    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        if "gmsh:physical" in mesh.cell_data:
            cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        else:
            cell_data = np.zeros(cells.shape[0], dtype=np.int32)
        points = mesh.points[:, :2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(
            points=points,
            cells={cell_type: cells},
            cell_data={"name_to_read": [cell_data.astype(np.int32)]},
        )
        return out_mesh

    if MPI.COMM_WORLD.rank==0:
        # Read in mesh
        msh = meshio.read(filename)

        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
        line_mesh = create_mesh(msh, "line", prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf", line_mesh)
    MPI.COMM_WORLD.barrier()

    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, name="Grid")

if mesh_type=='square':

    N = 8
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.triangle)
    
tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, 0)

imap_cells = mesh.topology.index_map(tdim)
imap_nodes = mesh.topology.index_map(0)

num_local_cells = imap_cells.size_local
num_local_nodes = imap_nodes.size_local

cells = mesh.topology.connectivity(tdim, 0).array.reshape((-1, 3))
nodes = mesh.geometry.x[:, :2]
cells_local = cells[:num_local_cells, :]
nodes_local = nodes[:num_local_nodes, :2]

topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh)
nodes = geometry[:, :2]
cells = topology.reshape((-1, 4))[:, 1:]

tri = Triangulation(nodes, cells) # Generate the triangulation

print(f'Rank {MPI.COMM_WORLD.rank} has {tri.Nnodes} nodes and {tri.Nelems} elements')

#ax.tripcolor(tri.tri_plt, facecolors=MPI.COMM_WORLD.rank * np.ones(tri.Nelems), edgecolors='black', cmap='viridis')

tri = MPI.COMM_WORLD.gather(tri, root=0)

if MPI.COMM_WORLD.rank==0:
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    cmap = 'viridis'
    norm = mpl.colors.Normalize(0, MPI.COMM_WORLD.size-1)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label='rank')
    for i in range(MPI.COMM_WORLD.size):
        facecolors = i*np.ones(tri[i].Nelems)
        ax.tripcolor(tri[i].tri_plt, facecolors=facecolors, edgecolors='black', cmap=cmap, norm=norm, alpha=0.5)

    plt.show()
