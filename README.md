Parallel Programming Course PHYS-743 at EPFL (2025)
The aim is to compute the mass matrix in parallel, for the finite elements method. We limit ourselves to $\mathbb P^1$ elements on triangulation, in 2D. 
We use dolfinx to obtain a partition of the mesh. To run on Helvetios, the partitions are saved to h5 files that are read on the cluster.
The ordering is unfortunately not the same as with dolfinx.
