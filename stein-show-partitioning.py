# Source : https://fenicsproject.discourse.group/t/plot-mesh-parallel-show-partitioning/16589 #

import pyvista, dolfinx
from mpi4py import MPI
import numpy as np

show_partitioning = True # or True

domain = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD,[(0,0),(1,1)], (10,10))
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(domain)
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
num_cells_local_geom = domain.geometry.index_map().size_local
num_dofs_per_cell = topology[0]

# Gather data
if show_partitioning:
    global_geometry = domain.comm.gather(geometry[:, :], root=0)
else:
    global_geometry = domain.comm.gather(geometry[:num_cells_local_geom, :], root=0)

    # Map topology to global dof indices
    topology_dofs = (np.arange(len(topology)) % (num_dofs_per_cell+1)) != 0
    global_dofs = domain.geometry.index_map().local_to_global(topology[topology_dofs].copy())
    topology[topology_dofs] = global_dofs

global_topology = domain.comm.gather(topology[: (num_dofs_per_cell + 1) * num_cells_local], root=0)
global_ct = domain.comm.gather(cell_types[:num_cells_local], root=0)

if domain.comm.rank == 0:
    root_geoms = [np.vstack(global_geometry)]  if not show_partitioning else global_geometry
    root_tops = [np.concatenate(global_topology)]  if not show_partitioning else global_topology
    root_cts = [np.concatenate(global_ct)]  if not show_partitioning else global_ct
    
    plotter = pyvista.Plotter()
    colors = ['k','r','b','m']
    for color, root_top, root_ct, root_geom in zip(colors,root_tops, root_cts, root_geoms):
        grid = pyvista.UnstructuredGrid(root_top, root_ct, root_geom)
        plotter.add_mesh(grid, style="wireframe", color=color, line_width=2)
    plotter.view_xy()
    plotter.show()
