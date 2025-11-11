### A Triangulation class ###
# Théophile Boinnard, updated 10/10/2025 #

### Import modules ###

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation as Triangulation_plt
from itertools import combinations
from functools import cached_property
from matplotlib.patches import Polygon
from collections import defaultdict
from matplotlib.animation import FuncAnimation
import copy
import meshio

### Class ###

class Triangulation:

    def __init__(self, nodes, elems) -> None:
        self.nodes = nodes
        self.elems = elems
        self.Nnodes = nodes.shape[0]
        self.Nelems = elems.shape[0]
        self._boundary = None

    @classmethod
    def from_file(cls, filename):
        mesh = meshio.read(filename)
        nodes = mesh.points[:, :2]
        elems = mesh.cells_dict['triangle']
        return cls(nodes, elems)

    def set_id(self, id):
        self._id = id

    def get_id(self):
        return self._id

    def copy(self):
        '''
        Returns a copy of the Triangulation instance.
        '''
        return copy.deepcopy(self)

    def set_boundary(self, B):
        '''
        Set the boundary, which is a boolean array, True for boundary points, False for internal points.
        '''

        if isinstance(B, np.ndarray) and B.shape[0]==self.Nnodes:
            self._boundary = B
        else:
            raise ValueError("The boundary should have as many values as the number of nodes.")

    def get_boundary(self):
        '''
        Get the boundary nodes.
        '''

        if self._boundary is None:
            raise ValueError("boundary is not set")

        return self._boundary

    @cached_property
    def edge_to_elem(self):
        '''
        For a given edge (that is, two nodes indices sorted in increasing order), returns the element(s) that have this edge (between 1 and 2.
        '''

        edge_to_elem_loc = defaultdict(list)

        # Iterate over elements and record the edges
        for elem_index, elem in enumerate(self.elems):
            # Iterate over combinations of two nodes (edges) in each element
            for edge in combinations(elem, 2):
                # Sort the edge tuple to ensure consistent ordering (e.g., (a, b) == (b, a))
                edge = tuple(sorted(edge))
                edge_to_elem_loc[edge].append(elem_index)

        return edge_to_elem_loc

    @cached_property
    def edges(self):
        '''
        The edges of the triangulation, that is the couples of two nodes indices, sorted in increasing order
        '''

        return np.array(list(self.edge_to_elem.keys()))

    @cached_property
    def Nedges(self):

        return self.edges.shape[0]

    @cached_property
    def neighbors(self):

        # Initialize the neighbor array with Nelems (default value for no neighbors)
        neigh = self.Nelems * np.ones_like(self.elems)

        # Iterate over the elements again to assign neighbors
        for elem_index, elem in enumerate(self.elems):
            for j, edge in enumerate(combinations(elem, 2)):
                edge = tuple(sorted(edge))
                # Find the neighboring elements that share this edge (excluding the current element)
                neighbor_elems = [e for e in self.edge_to_elem[edge] if e != elem_index]
                # If a neighbor exists, assign it, otherwise leave it as Nelems
                if neighbor_elems:
                    neigh[elem_index, j] = neighbor_elems[0]  # Each edge can only have one neighbor in 2D

        # Reorder neighbors to maintain compatibility with the original implementation (used for the mesh intersection code)
        return neigh[:, [0, 2, 1]]

    def plane_equations(self, z):
        '''
        Computes the coefficients that give the equations on the planes on each elements, for a given height z. z = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        '''

        M = np.concatenate((self.nodes[self.elems, :], np.ones((self.Nelems, 3, 1))), axis=2)
        coeffs = np.linalg.solve(M, z[self.elems])

        return coeffs

    @cached_property
    def elem_surf(self):
        '''
        Computes the surface of each element.
        '''

        surf = 0.5 * np.abs(np.cross(self.nodes[self.elems][: , 1, :] - self.nodes[self.elems][:, 0, :], self.nodes[self.elems][:, 2, :] - self.nodes[self.elems][:, 0, :]))

        return surf

    @cached_property
    def elem_edge_length(self):
        '''
        Computes the lenght of each edge of each element.
        '''

        edge_length = np.sqrt(np.sum((self.nodes[self.elems] - np.roll(self.nodes[self.elems], 2, axis=1))**2, axis=-1))

        return edge_length

    @cached_property
    def elem_diam(self):
        '''
        Computes the diameter of each element.
        '''

        diam = np.max(self.elem_edge_length, axis=1)

        return diam

    @cached_property
    def elem_inner_diam(self):
        '''
        Computes the inner diameter of each element.
        '''

        inner_diam = 2*self.elem_surf / np.sum(self.elem_edge_length, axis=1)

        return inner_diam

    @cached_property
    def elem_aspect_ratio(self):
        '''
        Computes the aspect ratio of each element.
        '''

        aspect_ratio = self.elem_diam / self.elem_inner_diam

        return aspect_ratio

    @cached_property
    def elem_edge_normal(self):
        '''
        Returns the normals of the edges (orientation is arbitrary).
        Order corresponding to elem_edge_length.
        '''

        sign_change = np.ones((self.Nelems, 3, 2))
        sign_change[:, :, -1] = -1
        n = np.flip(self.nodes[self.elems] - np.roll(self.nodes[self.elems], 2, axis=1), axis=-1) * sign_change
        n /= self.elem_edge_length[:, :, None]

        return n

    @cached_property
    def linear_maps(self):
        '''
        The reference element is the triangle (0,0) - (1,0) - (0,1)
        '''

        tK = self.nodes[self.elems][:, 0, :]
        MK = np.zeros((self.Nelems, 2, 2))
        MK[:, :, 0] = self.nodes[self.elems][:, 1, :] - self.nodes[self.elems][:, 0, :]
        MK[:, :, 1] = self.nodes[self.elems][:, 2, :] - self.nodes[self.elems][:, 0, :]

        return MK, tK

    @cached_property
    def linear_maps_svd(self):
        '''
        Computes the SVD of the jacobians of the linear maps.
        '''

        R, S, P = np.linalg.svd(self.linear_maps[0])

        return R, S, P

    @cached_property
    def nodes_to_elem(self):
        '''
        For each node, associate the elements that contain this node
        '''
        node_to_elements = {i: [] for i in range(self.Nnodes)}

        for element_index, element in enumerate(self.elems):
            for node in element:
                node_to_elements[node].append(element_index)

        return node_to_elements

    @cached_property
    def nodes_to_elem_array(self):
        '''
        Numba does not support dicts or lists of lists. This function converts tri.nodes_to_elem into a numpy array of fixed size, that is len(tri.nodes_to_elem) lines and (1+max([len(l) for l in tri.nodes_to_elem.values()]) columns. The first element of each column indicates the number of interesting value in each line.
        Input :
        - self : The triangulation
        Ouput :
        - nodes_to_elem_array : The property nodes_to_elem converted into an array
        '''

        nodes_to_elem_arr = np.empty((self.Nnodes, 1+max([len(l) for l in self.nodes_to_elem.values()])), dtype=int) # Set the array to the desired dimensions

        for i in range(self.Nnodes): # Loop over vertices

            nodes_to_elem_arr[i, 0] = len(self.nodes_to_elem[i]) # First element of each line is the number of useful element in line
            nodes_to_elem_arr[i, 1:nodes_to_elem_arr[i, 0]+1] = self.nodes_to_elem[i] # Set the values

        return nodes_to_elem_arr

    def longest_edge(self, i):
        '''
        From a node i, computes the longest of the elements that contain i.
        '''

        d = np.max(np.linalg.norm(self.nodes[self.elems[np.where(self.elems==i)[0]]] - self.nodes[i], axis=-1))

        return d

    def ball(self, i, r, eps = 1e-10):
        '''
        Returns all the nodes contained in the ball of radius r+eps, centred at node i.
        The eps parameter is added to account for numerical error.
        '''

        j = np.where(np.linalg.norm(self.nodes - self.nodes[i], axis=-1) <= r + eps)[0]

        return j

    @cached_property
    def reference_patch(self):
        '''
        Computes all the reference patches. That is, for each element K, find all the elements that share a vertex with K, K included.
        '''

        rp = []
        for i in range(self.Nelems):
            trp = [] # Elements in reference patch of element i
            for j in self.elems[i]:
                trp += self.nodes_to_elem[j] # Add elements that contains the node j
            trp = list(dict.fromkeys(trp)) # Remove duplicate
            rp.append(trp)

        return rp

    def connected_nodes(self, i):
        '''
        For node i, find all the nodes j connected to i.
        '''

        cn = np.unique(self.elems[np.where(self.elems==i)[0]])
        cn = cn[cn!=i] # remove i, it is not needed

        return cn

    ### Display functions ###

    @cached_property
    def tri_plt(self):

        return Triangulation_plt(self.nodes[:, 0], self.nodes[:, 1], self.elems)

    def plot(self, lw=1):
        '''
        Display the mesh.
        '''

        fig, ax = plt.subplots()

        ax.triplot(self.tri_plt, color='black', linewidth=lw)
        ax.set_aspect('equal')

        plt.show()

    def disp_nodes(self, z, dim='3D', show_mesh=True):
        '''
        Display qunatity z defined on nodes.
        '''

        if dim=='2D':

            fig, ax = plt.subplots()
            c = ax.tricontourf(self.tri_plt, z, cmap='viridis')
            if show_mesh:
                ax.triplot(self.tri_plt, color='black', linewidth=0.5)
            cb = plt.colorbar(c)
            ax.set_aspect('equal')
            plt.show()

        elif dim=='3D':

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlim(np.min(z), np.max(z))
            ax.set_xlim(np.min(self.nodes[:, 0]), np.max(self.nodes[:, 0]))
            ax.set_ylim(np.min(self.nodes[:, 1]), np.max(self.nodes[:, 1]))

            ax.plot_trisurf(self.tri_plt, z, cmap='viridis')
            plt.show()

        else:

            print('Uncorrect value for dim')

    def disp_dofs(self, dofs, z, dim='3D', show_mesh=True):
        '''
        Display qunatity z defined on given dofs.
        '''

        if dim=='2D':

            fig, ax = plt.subplots()
            c = ax.tricontourf(dofs[:, 0], dofs[:, 1], z, cmap='viridis')
            if show_mesh:
                ax.triplot(self.tri_plt, color='black', linewidth=0.5)
            cb = plt.colorbar(c)
            ax.set_aspect('equal')
            plt.show()

        elif dim=='3D':

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlim(np.min(z), np.max(z))
            ax.set_xlim(np.min(self.nodes[:, 0]), np.max(self.nodes[:, 0]))
            ax.set_ylim(np.min(self.nodes[:, 1]), np.max(self.nodes[:, 1]))

            ax.plot_trisurf(dofs[:, 0], dofs[:, 1], z, cmap='viridis')
            plt.show()

        else:

            print('Uncorrect value for dim')

    def disp_elem(self, z, dim='2D', show_mesh=True):
        '''
        Display quantity z defined on elements.
        '''

        if dim=='2D':

            fig, ax = plt.subplots()

            if show_mesh:
                c = ax.tripcolor(self.tri_plt, facecolors=z, edgecolors='black', cmap='viridis')
            else:
                c = ax.tripcolor(self.tri_plt, facecolors=z, cmap='viridis')
            plt.colorbar(c)
            ax.set_aspect('equal')

            plt.show()

        elif dim=='3D':

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlim(np.min(z), np.max(z))
            ax.set_xlim(np.min(self.nodes[:, 0]), np.max(self.nodes[:, 0]))
            ax.set_ylim(np.min(self.nodes[:, 1]), np.max(self.nodes[:, 1]))

            c = ax.plot_trisurf(self.tri_plt, z, cmap='viridis') #, edgecolor='black')
            cbar = plt.colorbar(c)

            plt.show()

        else:

            print('Uncorrect value for dim')

    ### Display patches ###

    def show_patch(self, i, j):
        '''
        Displays the node i and the patch of elements generated by the nodes j.
        '''

        fig, ax = plt.subplots()

        ax.triplot(self.tri_plt, color='black')
        ax.plot(self.nodes[j, 0], self.nodes[j, 1], 'ro')
        ax.plot(self.nodes[i, 0], self.nodes[i, 1], 'b*')
        ax.set_aspect('equal')

        plt.show()

    def show_patch_dofs(self, dofs, i, j):
        '''
        Displays the vertex i and the dofs j. Used to show the patch for Naga-Zhang post-processing.
        '''

        fig, ax = plt.subplots()

        ax.triplot(self.tri_plt, color='black')
        ax.plot(self.nodes[i, 0], self.nodes[i, 1], 'ro')
        ax.plot(dofs[j, 0], dofs[j, 1], 'b*')
        ax.set_aspect('equal')

        plt.show()

    def show_reference_patch(self, i):
        '''
        Displays the reference patch of elements i.
        '''

        fig, ax = plt.subplots()

        ax.triplot(self.tri_plt, color='black')
        ax.add_patch(Polygon(self.nodes[self.elems[i]], color='blue', alpha=0.5))
        for j in self.reference_patch[i]:
            ax.add_patch(Polygon(self.nodes[self.elems[j]], color='red', alpha=0.5))

        ax.set_aspect('equal')

        plt.show()

    ### Animation functions ###

    def anim_nodes(self, sol, dim='3D', show_mesh=True, save=False):
        '''
        Makes an animation that display sol['uh'] in time.
        '''

        if dim!='2D' and dim!='3D':

            raise ValueError('Please insert a correct value of dim (either 2D or 3D).')

        min_uh = np.min([np.min(uh.x.array) for uh in sol['uh']])
        max_uh = np.max([np.max(uh.x.array) for uh in sol['uh']])

        if dim=='3D':

            # Create the initial plot
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.set_zlim(min_uh, max_uh)

            # Function to update the plot for each frame
            def update(frame):
                ax.clear()  # Clear previous plot
                ax.plot_trisurf(self.tri_plt, sol['uh'][frame].x.array, cmap='viridis', edgecolor='none', vmin=min_uh, vmax=max_uh)
                ax.set_title(f't = {sol["t"][frame]:.2f}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlim(min_uh, max_uh)

        elif dim=='2D':

            fig, ax = plt.subplots()
            ax.set_xlim(self.nodes[:, 0].min(), self.nodes[:, 0].max())
            ax.set_ylim(self.nodes[:, 1].min(), self.nodes[:, 1].max())

            tripcol = ax.tripcolor(self.tri_plt, sol['uh'][0].x.array, cmap='viridis', vmin=min_uh, vmax=max_uh)

            if show_mesh:
                ax.triplot(self.tri_plt, color='black', linewidth=0.5)
            ax.set_aspect('equal')

            # Add a colorbar once, based on the initial tripcolor
            cbar = fig.colorbar(tripcol, ax=ax)

            # Function to update the plot for each frame
            def update(frame):
                # Update the tripcolor plot with new data
                for collection in ax.collections:
                    collection.remove()
                # Draw a new tripcolor plot with the updated values
                ax.tripcolor(self.tri_plt, sol['uh'][frame].x.array, cmap='viridis', vmin=min_uh, vmax=max_uh)
                ax.set_title(f't = {sol["t"][frame]:.2f}')
                if show_mesh:
                    ax.triplot(self.tri_plt, color='black', linewidth=0.5)

        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(sol['uh']), interval=100)

        if save:
            anim.save('finite_element_solution.mp4', writer='ffmpeg')

        plt.show()

    def anim_elems(self, z, t=None, show_mesh=True, save=False):
        '''
        Animate quantity z defined on the elements in time.
        '''

        fig, ax = plt.subplots()
        ax.set_xlim(self.nodes[:, 0].min(), self.nodes[:, 0].max())
        ax.set_ylim(self.nodes[:, 1].min(), self.nodes[:, 1].max())

        min_z = np.min(z)
        max_z = np.max(z)

        edgecolors = 'black' if show_mesh else None
        c = ax.tripcolor(self.tri_plt, facecolors=z[:, 0], edgecolors=edgecolors, cmap='viridis', vmin=min_z, vmax=max_z)
        ax.set_aspect('equal')

        # Add a colorbar once, based on the initial tripcolor
        cbar = fig.colorbar(c, ax=ax)

        # Function to update the plot for each frame
        def update(frame):
            # Update the tripcolor plot with new data
            for collection in ax.collections:
                collection.remove()
            # Draw a new tripcolor plot with the updated values
            c = ax.tripcolor(self.tri_plt, facecolors=z[:, frame], edgecolors=edgecolors, cmap='viridis', vmin=min_z, vmax=max_z)
            ax.set_title(f't = {t[frame]:.2f}') if t else None

        # Create the animation
        anim = FuncAnimation(fig, update, frames=z.shape[1], interval=100)

        if save:
            anim.save('finite_element_solution.mp4', writer='ffmpeg')

        plt.show()

    def anim_dofs(self, sol, dim='3D', show_mesh=True, save=False):
        '''
        Makes an animation that display sol['uh'] in time, when uh is more than P1
        '''

        V = sol['uh'][0].function_space
        dofs = V.tabulate_dof_coordinates()[:, :2]

        if dim!='2D' and dim!='3D':

            raise ValueError('Please insert a correct value of dim (either 2D or 3D).')

        min_uh = np.min([np.min(uh.x.array) for uh in sol['uh']])
        max_uh = np.max([np.max(uh.x.array) for uh in sol['uh']])

        if dim=='3D':

            # Create the initial plot
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.set_zlim(min_uh, max_uh)

            # Function to update the plot for each frame
            def update(frame):
                ax.clear()  # Clear previous plot
                ax.plot_trisurf(dofs[:, 0], dofs[:, 1], sol['uh'][frame].x.array, cmap='viridis', vmin=min_uh, vmax=max_uh)
                ax.set_title(f't = {sol["t"][frame]:.2f}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlim(min_uh, max_uh)

        elif dim=='2D':

            fig, ax = plt.subplots()
            ax.set_xlim(self.nodes[:, 0].min(), self.nodes[:, 0].max())
            ax.set_ylim(self.nodes[:, 1].min(), self.nodes[:, 1].max())

            tripcol = ax.tricontourf(dofs[:, 0], dofs[:, 1], sol['uh'][0].x.array, cmap='viridis', vmin=min_uh, vmax=max_uh)
            if show_mesh:
                ax.triplot(self.tri_plt, color='black', linewidth=0.5)
            ax.set_aspect('equal')

            # Add a colorbar once, based on the initial tripcolor
            cbar = fig.colorbar(tripcol, ax=ax)

            # Function to update the plot for each frame
            def update(frame):
                # Update the tripcolor plot with new data
                for collection in ax.collections:
                    collection.remove()
                # Draw a new tripcolor plot with the updated values
                ax.tricontourf(dofs[:, 0], dofs[:, 1], sol['uh'][frame].x.array, cmap='viridis', vmin=min_uh, vmax=max_uh)
                ax.set_title(f't = {sol["t"][frame]:.2f}')
                if show_mesh:
                    ax.triplot(self.tri_plt, color='black', linewidth=0.5)

        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(sol['uh']), interval=100)

        if save:
            anim.save('finite_element_solution.mp4', writer='ffmpeg')

        plt.show()
