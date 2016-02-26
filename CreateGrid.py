""" Create a 2D array, where the cells contain the neutral Hydrogen column density seen by the cell.
    The array is saved to the location specified by "save_name", for later use.
    Requirements:
        PhysicalConstants.py                     (packaged with this)
        SnapshotParameters.py                    (packaged with this)
        MiscellaneousFunctions.py                (packaged with this)
        Eagle snapshot in .hdf5 format
        Urchin run on Eagle snapshot in .hdf5 format
        Numpy module
        Scipy module
        H5py module
        os module
    To use:
        Change all parameters under "SET ME !!!" heading to ones specific to your system.
        Run this file.
    Algorithm:
        Define M grids of different sizes   (use lower resolution grids for larger particles to speed up computation)
        Calculate column densities:
            For each particle, find the cells it contributes too.
            For those cells, calculate the amount of particle that overlaps with them.
            Apply periodic boundary conditions to the cells.
            Add the cells back to grid.
        Compile the grids into a single grid.
    This file is written by Matt Law and is provided "as is". """

from __future__ import division
import os
import scipy.special
import numpy as np
import h5py
from PhysicalConstants import *         # Supplies all relevant physical constants in a 'Constant' object
from SnapshotParameters import *        # Supplies all relevant snapshot parameters in a 'Snapshot' object
from MiscellaneousFunctions import *    # Provides some useful 'accessory' functions
__author__ = 'Matt Law'

# SET ME !!! ********************************************************************************************************* #
snapshot_folder = ''        # The directory containing the EAGLE snapshot
urchin_folder   = ''        # The directory containing the urchin snapshot
save_name = "grid1"         # The name to save the final grid with
n_grids = 3                 # The number of different grids to use
min_res = 128               # Minimum number of cells in one side of the smallest grid. I.e. grid is of size 512x512
max_res = 4096              # Maximum number of cells in one side of the biggest grid.  I.e. grid is of size 4096x4096


# FUNCTIONS ********************************************************************************************************** #
# 1. Read in EAGLE snapshot ========================================================================================== #
def import_hdf5_file(file_path='.', part_type=0, hdf_paths=['Coordinates','Density','SmoothingLength']):
    """ Imports information from a single HDF5 file.
    :param file_path: A string containing the complete HDF5-file location, e.g. 'C:/Users/Matt/Documents/Uni Work 4/Project/Eagle Simulations/snap_028_z000p000'
    :param part_type: The type of particles to read information about: 0 gas, 1 dark matter, 4 stars, 5 black holes (2 and 3 are non-existent)
    :param hdf_paths: A list of strings, where each string is a name of a data folder in the hdf5 file.
    :return: A list of arrays, where each array is the data in the folder specified by the corresponding hdf_paths string. """
    f = h5py.File(file_path, 'r')
    data = []
    for i in hdf_paths:
        data.append(np.array(f['PartType'+str(part_type) + '/' + i]))
    f.close()
    return data


def import_hdf5_folder(folder_path='.', part_type=0, hdf_paths=['Coordinates','Density','SmoothingLength']):
    """ Import information from all HDF5 files in a folder
    :param folder_path: A string containing the folder with all the HDF5-files in, e.g. 'C:/Users/Matt/Documents/Uni Work 4/Project/Eagle Simulations'
    :param part_type: The type of particles to read information about: 0 gas, 1 dark matter, 4 stars, 5 black holes (2 and 3 are non-existent)
    :param hdf_paths: A list of strings, where each string is a name of a data folder in the hdf5 file.
    :return: A list of arrays, where each array is the data in the folder specified by the corresponding hdfPaths string.
    """
    dirs = os.listdir(folder_path)
    data = import_hdf5_file(folder_path + '/' + dirs[0], part_type=part_type, hdf_paths=hdf_paths)
    for i, fileName in enumerate(dirs[1:-1]):
        new_data = import_hdf5_file(folder_path + '/' + fileName, part_type=part_type, hdf_paths=hdf_paths)
        for j in range(len(new_data)):
            data[j] = np.concatenate((data[j], new_data[j]))
    return data


# 2. Define the grid(s) ============================================================================================== #
def define_grids(min_res, max_res, num_res, h):
    if num_res > 1:
        q = np.arange(rd_to_int(np.log2(min_res)), rd_to_int(np.log2(max_res))+1, step=(((np.log2(max_res))-(np.log2(min_res)))/(num_res-1)))
        w = q.astype(int)
        for i in range(len(q)):
            if float(w[i]) != q[i]:
                w[i] += 1

        grid_res = 2**w                                         # A list of the grid resolutions (all are a power of 2)
        grid_res_float=grid_res.astype(float)
        cell_len = Snapshot.BoxsizecMpch/grid_res_float         # A list containing the cell length of each grid (comoving sim. units)
        grids    = [np.zeros((i, i)) for i in grid_res]         # A list of the grids of corresponding resolutions
        h_bounds = np.linspace(min(h), max(h), num_res+1)       # A list of boundaries for h, which will select which grid is used for a particle
        h_bounds = h_bounds[1:len(h_bounds)]                    # Stops the highest resolution only being for 1 particle. The +1 in the above line counteracts this line.

        grid_h   = np.zeros(len(h), dtype=int)                  # A list where each element corresponds to a particle's grid
        for i in xrange(len(h)):                                # Iterate over every particle
            for j in xrange(len(h_bounds)-1):                   # Assume h_bounds is in ascending order
                if h[i] <= h_bounds[j]:                         # If particles size (h[i]) < boundary, store index of grid corresponding to boundary, j.
                    grid_h[i] = abs(j-(len(h_bounds)-1))
                    break
    else:                                                       # Assume that num_res = 1
        grid_res = max_res
        grids    = [np.zeros((grid_res, grid_res), dtype=float)]
        cell_len = [Snapshot.BoxsizecMpch/grid_res]
        h_bounds = [max(h)]
        grid_h = np.zeros(len(h), dtype=int)
    return grids, cell_len, grid_h, h_bounds


# 3. Calculate column densities ====================================================================================== #
def integrate_gauss_erf(h, x0):
    """ Uses error functions to find the integral of a Gaussian. Returns a function. """
    pi = np.pi
    t = 2*pi**(1/6)
    A = t/h             # = sqrt(A) of Gaussian function
    N = ((pi**(1/2))/(2*A))*(( 8/(pi*(h**3)) ) / ( scipy.special.erf(t) - 2*t*np.exp(-t*t)/np.sqrt(pi) ))**(1/3) # Have combined the erf's coefficients with N for speed up.
    x0= x0
    def erf(lim1, lim2):
        return N*(scipy.special.erf(A*(lim2-x0)) - scipy.special.erf(A*(lim1-x0)))  # 2/sqrt(pi)*integral(exp(-t**2), t=0..z).
    return erf


def calculate_column_densities_express(grids, x, y, h, particle_parameters, cell_len, grid_h):
    """ See non-express version for details. Employs the symmetry of Gaussians to speed up computation.
        Assumes Gaussian components are the same along the x and y."""

    # Combine all particle parameters into a single array, for easy indexing inside the main loop.
    if particle_parameters == []:
        A = np.ones(len(x))
    else:
        A = np.ones(len(particle_parameters[0]))
        if len(particle_parameters) == 1:
            A *= particle_parameters[0]
        else:
            for l in xrange(len(particle_parameters[0])):
                for m in xrange(len(particle_parameters)):
                    A[l] *= particle_parameters[m][l]

    for k in xrange(len(h)):
        cLen    = cell_len[grid_h[k]]                                                       # cLen = size of cell in grid
        cGrid   = grids[grid_h[k]]                                                          # Pick the correct grid for particle

        i_x     = int(x[k]/cLen)                                                            # Index of cell central to particle
        i_y     = int(y[k]/cLen)
        N_off   = int(np.ceil(h[k]/cLen))                                                   # Width of particles in cells

        x_sub_grid = np.arange(i_x - N_off, i_x + N_off, 1)                                 # A square sub-grid that just encompasses particle
        x_comp     = integrate_gauss_erf(h[k],x[k])                                         # Pre-computation/Closure of Gaussian mass distribution of particle
        x_comps    = [x_comp(i*cLen, (i+1)*cLen) for i in x_sub_grid]                       # Integrated component of particles mass for all cells along the x-row of sub-grid.

        y_sub_grid = x_sub_grid - i_x + i_y
        x_sub_grid = x_sub_grid % (cGrid.shape[0])                                          # Apply Periodic boundary conditions. N.B. Must be after calculate of x-comps!
        y_sub_grid = y_sub_grid % (cGrid.shape[1])
        x_sub_grid = x_sub_grid[:, np.newaxis]                                              # See for fancy indexing use: http://stackoverflow.com/questions/4257394/slicing-of-a-numpy-2d-array-or-how-do-i-extract-an-mxm-submatrix-from-an-nxn-ar
        particle_grid = np.array(x_comps)[:, np.newaxis] * np.array(x_comps)
        particle_grid *= A[k] / np.sum(particle_grid)                                       # Renormalise to correct for errors in mass
        cGrid[y_sub_grid, x_sub_grid] = cGrid[y_sub_grid, x_sub_grid] + particle_grid
        grids[grid_h[k]] = cGrid

    return grids


# 4. Combine grids =================================================================================================== #
def compile_grids(grids, cell_len, h_bounds):
    """ Take a list of 2D arrays, 'grids', and compile them into a single grid. """
    h_bounds_cell_units = np.zeros(len(h_bounds))
    for i in range(len(h_bounds)):
        h_bounds_cell_units[i] = h_bounds[i]/cell_len[-i-1]

    if len(grids) > 1:
        for i in range(len(grids)-1):                                                   # Iterate over the list of grids. For each: split the smaller grid into
            ratio = grids[1].shape[0]/grids[0].shape[0]                                 # the larger grid's number of cells (divide component by number of cells split into - to keep total mass constant)
            grids[1] = grids[1] + (np.repeat(np.repeat(grids[0], ratio, axis=0), ratio, axis=1)/(ratio*ratio))
            del grids[0]                                                                # List gets 1 shorter, so now indexes refer to a new element
    return grids[0], min(cell_len)


# 5. Convert units =================================================================================================== #
def convert_cell_to_cgs(grid, cLen):
    pLen = cLen * Snapshot.ExpansionFactor * Snapshot.U_L/Snapshot.HubbleParam  # Physical cell length          :: cm
    grid = grid * (Snapshot.U_M/Snapshot.HubbleParam) / (pLen**2)               # Convert to physical CGS       :: g/cm^2
    grid = grid/Constant.m_HI                                                   # Convert to number density     :: Num/cm^2
    return grid, cLen


# 6. Save grid ======================================================================================================= #
def save_grid(array, file_name):
    np.save(file_name, array)


# 7. Plot grid ======================================================================================================= #
def plot_grid(grid, save_name, colour_map_splits=[17.2, 20.3], halos=False, halo_x=None, halo_y=None, halo_radii=None, extra_close=False):
    cell_to_pkpc = Snapshot.Boxsizepkpc/grid.shape[0]

    extent  = (0, grid.shape[0]*cell_to_pkpc, 0, grid.shape[1]*cell_to_pkpc)
    x_units = r"$ \rm pkpc$"

    if np.min(grid) <= 0:
        grid = ma.masked_where(grid <= 0, grid)
    colour_map = colour_map_split3(np.log10(grid), colour_map_splits[0],colour_map_splits[1])

    fig = plt.figure()
    plt.imshow(np.log10(grid), origin='lower', interpolation='none', cmap = colour_map, extent=extent)
    cbar=plt.colorbar(pad=0)
    cbar.set_label(r"$ \rm log_{10} \ (N_{HI}/cm^{-2}) $", rotation=-90, labelpad=15)
    plt.xlabel(r"$ \rm x$"+"$ \ ($"+x_units+"$)$", labelpad=3)
    plt.ylabel(r"$ \rm y$"+"$ \ ($"+x_units+"$)$", labelpad=2)

    if halos:
        for i in xrange(0, len(halo_radii)):
            circle = plt.Circle((halo_x[i], halo_y[i]), radius=halo_radii[i],
                                facecolor="none", edgecolor='black', alpha=0.7, linestyle='solid', linewidth=1)
            fig.gca().add_artist(circle)

    plt.savefig(save_name+".pdf", dpi=1000, bbox_inches='tight', pad_inches=0.04)
    plt.savefig(save_name+".png", dpi=300 , bbox_inches='tight', pad_inches=0.04)
    plt.savefig(save_name+".jpg", dpi=300 , bbox_inches='tight', pad_inches=0.04)

    # Close up of rightmost over density
    if extra_close:
        plt.xlim((2880,3020))
        plt.ylim((780, 920))

    else:
        plt.xlim((2700,3100))
        plt.ylim((700, 1100))
    plt.savefig(save_name+"CloseUp.pdf", dpi=1000, bbox_inches='tight', pad_inches=0.04)
    plt.savefig(save_name+"CloseUp.png", dpi=300 , bbox_inches='tight', pad_inches=0.04)
    plt.savefig(save_name+"CloseUp.jpg", dpi=300 , bbox_inches='tight', pad_inches=0.04)
    plt.show()


# RUN FUNCTIONS ****************************************************************************************************** #
if __name__ == '__main__':
    # 0. Select the correct snapshot and import it's relevant parameters ============================================= #
    Snapshot.importSnap(snapshot_folder, urchin_folder)

    # 1. Import information from Snapshot ============================================================================ #
    coordinates, h, m, x_hi = import_hdf5_folder(Snapshot.folder_name, 0, hdf_paths=['Coordinates', 'SmoothingLength', 'Mass', 'SmoothedElementAbundance/Hydrogen'])
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    coordinates = None
    del coordinates
    nf = import_hdf5_folder(Snapshot.urchin_name, hdf_paths=['HydrogenOneFraction'])[0]

    # 2. Create grid ================================================================================================= #
    grids, cell_lengths, grid_h, h_bounds = define_grids(min_res, max_res, n_grids, h)
    print "Grids created."

    # 3. Calculate Column densities ================================================================================== #
    grids = calculate_column_densities_express(grids, x, y, h, [m, x_hi, nf], cell_lengths, grid_h)
    print "Column Densities calculated."

    # 4. Compile grids =============================================================================================== #
    grid, cell_length = compile_grids(grids, cell_lengths, h_bounds)
    print "Grids compiled."

    # 5. Convert units =============================================================================================== #
    grid, cell_length = convert_cell_to_cgs(grid, cell_length)
    print "Units converted."

    # 6. Saving grid ================================================================================================= #
    save_grid(grid, save_name)
    print "Grid saved."

    # 7. Plot grid =================================================================================================== #
    plot_grid(grid,save_name)
