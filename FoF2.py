"""
    Runs a Friends Of Friends algorithm on a discrete 2D grid.
    It groups neighbouring cells, and returns a list of groups.
    The result of this is used to estimate the area of LLSs.
"""

from __future__ import division
from scipy import integrate
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as random
import os
import scipy.special
import time
import numpy.ma as ma
import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter, LogFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.ticker import FuncFormatter
from MiscellaneousFunctions import *
from SnapshotParameters import *
import ProjectFiles
from Subfind_file import *
from CorrelationFunctions import *

__author__    = 'Matt'

# Set me ============================================================================================================= #
file_id     = 7                                     # A unique id to save graphs and files with
save_folder = "FoF/snap_012_z003p017/"
grid_folder = save_folder
grid_name   = "3NGrids3RMin128RMax4096NDGrid.npy"   # "3NGrids3RMin128RMax2048D0.001Grid.npy"

#FONTSIZE = 12                                      # Set the size of the font in graph labels for figures that will span the whole page
dpi         = 100                                   # Dots-per-inch: high for high quality save figures (1200), low for quick figures (100)
use_fake    = 0                                     # Use fake data for debugging and testing
load_grps   = 1                                     # 1 load groups from file with name 'load_name'. 0 calculate groups
load_name   = "Groups1200v2.npy"
save_grps   = 0                                     # 0 don't save groups, 1 save groups to file with name 'save_name'
save_name   = "Groups1200v2"

PLOT_GRID                   = 0                     # Produce a plot of the loaded grid
PLOT_GROUPS_COLOUR_MAP      = 0                     # Produce a plot where the groups are coloured differently
PLOT_GROUPS_SCATTER         = 0                     # Same as above, but as a scatter plot

PLOT_HISTOGRAM_SURFACE_AREA = 0                     # Plot a histogram of the surface areas of LLSs
FRACTION_AREA_IN_LLS        = 0                     # Plot the above, but as a fraction of total area

small_fig_width_pt = 223.51561 + 18 # Get this from LaTeX using \showthe\columnwidth
big_fig_width_pt = 526.72534 + 25

fig_width_pt = small_fig_width_pt
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = 0.772 #5**0.5-1.0)/2.0    # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_size = [fig_width,fig_height]

params = {                              # pyplot parameters (tweak as desired)
        'backend'           : 'ps',
        'mathtext.default'  : 'regular',
        'font.family'       : 'Times',
        'font.size'         : 11,
        'font.weight'       : 400,      #'bold',
        'axes.labelsize'    : 11,
        'legend.fontsize'   : 11,
        'xtick.labelsize'   : 8,
        'ytick.labelsize'   : 8,
        #'mathtext.fontset' : 'custom',
        #'mathtext.cal'     : 'Times',
        #'mathtext.tt'      : 'Times',
        #'mathtext.rm'      : 'Times',
        #'mathtext.bf'      : 'Times:bold' ,
        'text.usetex'       : False,
        'text.latex.unicode': True,
        'figure.figsize'    : fig_size
        }
plt.rcParams.update(params)

# Global constants        ============================================================================================ #
# Physical constants
pi      = np.pi                 # pi                        :: unitless
c       = 2.99792458e10         # Speed of light            :: cm/s
M0      = 1.9891e33             # Sun's mass                :: g
L_sun   = 3.846e33              # Sun's Luminosity          :: gcm^2s^-3
G       = 6.67384e-8            # Gravitational constant    :: cm3/g/s2
Mpc     = 3.08567758e24         # 1 Mpc                     :: cm
planck  = 6.62606957e-34        # Planck's Constant         :: m^2.kg/s
m_p     = 1.67262178e-24        # Mass of proton            :: g
m_e     = 9.10938291e-28        # Mass of electron          :: g
m_HI    = 1.6737236e-24         # Mass of neutral Hydrogen  :: g

# Simulation parameters (hardcoded for now)
U_L     = 3.085678E24           # Unit Length - 1 Mpc       :: cm
U_M     = 1.989E43              # Unit Mass - 10^10*M_sun   :: g
H_0     = 67.77 * 1000 * 100    # Hubble constant (planck)  :: cms^-1Mpc^-1
HubbleC = 0.6777                # Hubble constant/100       :: unitless
a       = 0.2489726990748467    # Expansion factor          :: unitless (1/1+z)
redshift= 3.0165046357126       # Redshift                  :: unitless
Hz      = 9.964708881888158E-18 # Hubble paramter           :: s^-1     (np.sqrt( ( (H_0/Mpc )**2 * (Omega0/(a**3) + OmegaLambda)) )
m       = 1.2252497e-4          # Mass of 1 particle        :: g*h/U_M
Omega0  = 0.307                 # Mass fraction of ...      :: unitless
OmegaBaryon = 0.0482519         # Baryon mass fraction      :: unitless
OmegaLambda = 0.693             # Dark energy mass fraction :: unitless

cboxsizeMpch = 8.47125                  # cMpc/h
cboxsizeMpc  = cboxsizeMpch/HubbleC     # cMpc
pboxsizeMpc  = cboxsizeMpc*a            # pMpc
pboxsize     = pboxsizeMpc*1000         # pkpc
print "Box size ", pboxsize, "pkpc"

# Calculated constants
p_c     = 3*(Hz**2)/(8*pi*G)                        # Critical density  :: g/cm^3


def save_name_generator(grid_name, file_id, use_fake):
    """ Generates a save name, based on the parameters of the grid. """
    save_name = grid_name[0:len(grid_name)-4]+str(file_id)
    if use_fake == 1:
        save_name = save_name + "TEST"
    return save_name


def greek_symbols():
    """ Returns a dictionary of the greek alphabet and their unicode expressions """
    greek_alphabet = {'tau': u'\u03c4', 'lamda': u'\u03bb', 'xi': u'\u03be', 'Chi': u'\u03a7', 'delta': u'\u03b4', 'omega': u'\u03c9', 'upsilon': u'\u03c5', 'Theta': u'\u0398', 'zeta': u'\u03b6', 'Pi': u'\u03a0', 'Nu': u'\u039d', 'Phi': u'\u03a6', 'Psi': u'\u03a8', 'Epsilon': u'\u0395', 'Omicron': u'\u039f', 'Iota': u'\u0399', 'Beta': u'\u0392', 'Rho': u'\u03a1', 'Delta': u'\u0394', 'Upsilon': u'\u03a5', 'pi': u'\u03c0', 'Omega': u'\u03a9', 'iota': u'\u03b9', 'phi': u'\u03c6', 'psi': u'\u03c8', 'Kappa': u'\u039a', 'epsilon': u'\u03b5', 'omicron': u'\u03bf', 'Mu': u'\u039c', 'beta': u'\u03b2', 'Eta': u'\u0397', 'rho': u'\u03c1', 'Alpha': u'\u0391', 'alpha': u'\u03b1', 'Sigma': u'\u03a3', 'Gamma': u'\u0393', 'Tau': u'\u03a4', 'Lamda': u'\u039b', 'Xi': u'\u039e', 'kappa': u'\u03ba', 'nu': u'\u03bd', 'mu': u'\u03bc', 'eta': u'\u03b7', 'chi': u'\u03c7', 'Zeta': u'\u0396', 'sigma': u'\u03c3', 'theta': u'\u03b8', 'gamma': u'\u03b3'}
    return greek_alphabet


greek = greek_symbols()

base_save_name = save_name_generator(grid_name, file_id, use_fake)

# ******************************************************************************************************************** #
# Code start ========================================================================================================= #

# [i,j]
#    j ------- >
#  i
#  |
#  |
# down
# Summary: Increasing i moves down, increasing j moves right

def flipud_groups(groups, shape):
    """
    :return: groups but with j component flipped
    """
    for i in range(len(groups)):
        x,y = zip(*groups[i])
        x2 = range(0, len(x))
        #print len(y), len(y2)
        for j in range(len(x)):
            x2[j] = shape[1] - x[j]
        groups[i] = zip(x2,y)
    return groups


def check_neighbours(grid, index):
    #print index
    i,j = index
    neighbours = []

    sx = grid.shape[0]
    sy = grid.shape[1]

    right   = ( (i+1) % sx, (j)   % sy )                                        # Define neighbouring cell's coordinates
    left    = ( (i-1) % sx, (j)   % sy )                                        # And apply periodic boundary conditions
    up      = ( (i)   % sx, (j+1) % sy )                                        #
    down    = ( (i)   % sx, (j-1) % sy )                                        #

    if grid[right] != np.ma.core.MaskedConstant:                                # If Pixel contains something
        neighbours.append(right)                                                # Store it's coordinates
        grid[right] = ma.masked                                                 # Mask value, so it won't be double counted later

    if grid[left]  != np.ma.core.MaskedConstant:
        neighbours.append(left )
        grid[left]  = ma.masked

    if grid[up]    != np.ma.core.MaskedConstant:
        neighbours.append(up   )
        grid[up]    = ma.masked

    if grid[down]  != np.ma.core.MaskedConstant:
        neighbours.append(down )
        grid[down]  = ma.masked

    return grid, neighbours


def FoF2(grid):

    groups = []
    # Isolate LL pixels
    grid = ma.masked_where(np.log10(grid) < 17.2, grid)                     # Isolate LLs Pixels
    grid = ma.masked_where(np.log10(grid) > 20.3, grid)

    g = 0                                                                   # Group Number (index of groups)
    while grid.count() != 0:                                                # Iterate until all pixels have been added to a group

        indexs = grid.nonzero()                                             # Select a pixel: Find all non-masked pixels
        i,j = indexs[0][0], indexs[1][0]                                    #                 and pick the first one.

        groups.append([(i,j)])                                              # append to group

        recent_list = [(i,j)]                                               # Do 1 group:

        while len(recent_list) != 0:                                        # Iterate until no more neighbouring pixels are found

            grid, neighbours = check_neighbours(grid, recent_list[0])       # Get the coordinates of the pixels that neighbour the pixel being looped over if they have a LLS in
            recent_list = recent_list+neighbours                            # Add all the new neighbours to recent_list
            del recent_list[0]                                              # Remove the pixel that is being looped over

            grid[i,j] = ma.masked
            if len(recent_list) > 0:
                groups[g].append(recent_list[0])                            # Add new pixel to master list of pixels ( groups ) that will be iterated over next

        g += 1                                                              # Finished 1 whole group, so start the next
        print "one group done", grid.count()

    return groups


def FoF2wGroupNums(grid, subhalo_x, subhalo_y, subhalo_mass):

    groups = []
    subhalo_masses = []
    subhalo_indexes = []
    # Isolate LL pixels
    grid = ma.masked_where(np.log10(grid) < 17.2, grid)                     # Isolate LLs Pixels
    grid = ma.masked_where(np.log10(grid) > 20.3, grid)

    g = 0                                                                   # Group Number (index of groups)
    while grid.count() != 0:                                                # Iterate until all pixels have been added to a group

        indexs = grid.nonzero()                                             # Select a pixel: Find all non-masked pixels
        i,j = indexs[0][0], indexs[1][0]                                    #                 and pick the first one.

        groups.append([(i,j)])                                              # append to group

        recent_list = [(i,j)]                                               # Do 1 group:

        while len(recent_list) != 0:                                        # Iterate until no more neighbouring pixels are found

            grid, neighbours = check_neighbours(grid, recent_list[0])       # Get the coordinates of the pixels that neighbour the pixel being looped over if they have a LLS in
            recent_list = recent_list+neighbours                            # Add all the new neighbours to recent_list
            del recent_list[0]                                              # Remove the pixel that is being looped over

            grid[i,j] = ma.masked
            if len(recent_list) > 0:
                groups[g].append(recent_list[0])                            # Add new pixel to master list of pixels ( groups ) that will be iterated over next


        # Find nearest subhalo and assume that the LLSs originate from it
        xs, ys =  zip(*groups[g])
        # Convert units of xs and ys to the same as subhalo_x and y
        xs = np.mean(xs)*cell_to_cMpch
        ys = np.mean(ys)*cell_to_cMpch

        diff = (xs - subhalo_x)**2 + (ys - subhalo_y)**2
        idx = np.argmin(diff)
        subhalo_masses.append(subhalo_mass[idx])
        subhalo_indexes.append(idx)

        g += 1                                                              # Finished 1 whole group, so start the next
        print "one group done", grid.count()

    return groups, subhalo_masses


def writeGroupsToArray(groups, grid):
    for i in xrange(len(groups)):
        #print groups[i]
        for j in groups[i]:
            grid[j] = i + 1
    return grid


def plot_maximal_extent(groups):

    # Find maximal diameters of LLSs
    # Approximate them as the maxmimum distance between 2 pixels in the LLSs
    # Find distances from origin for all LLS pixels in groups
    print groups

    distances = []
    for i in xrange(len(groups)):
        LLS = groups[i]
        LLS_distances = []
        for j in xrange(len(LLS)):
            d1 = (LLS[j][0]**2 + LLS[j][1]**2)**0.5
            for k in xrange(len(LLS)):
                d2 = (LLS[k][0]**2 + LLS[k][1]**2)**0.5
                LLS_distances.append( d1 - d2 )
        distances.append(max(LLS_distances))
    np.save("FoF/MaximalExtent/distances", distances)

    center, hist = histogram(distances)

    plt.figure()
    plt.semilogx()
    plt.scatter(np.log10(center), hist)
    plt.xlabel(r"$ \rm Maximal size (pkpc) $")
    plt.ylabel(r"$ \rm Number of LLSs $")
    plt.show()


def overlap(grid1, grid2):
    """ Measures the overlap between 2 grids. """
    
    # 1.1. Iterate over grid1
    idxs_i, idxs_j = grid1.nonzero()
    print idxs_i
    number_of_potential_overlaps = len(idxs_i)
    num_overlaps = 0
    for i in xrange(0, len(idxs_i), 1):
        if grid1[idxs_i[i], idxs_j[i]] > 17.2 and grid1[idxs_i[i], idxs_j[i]] < 20.3:       # LLS
            if grid2[idxs_i[i], idxs_j[i]] > 17.2 and grid2[idxs_i[i], idxs_j[i]] < 20.3:   # LLS
                num_overlaps += 1
    print num_overlaps
    
                
def mass_subhaloVsNumberLLS():
    """ Plot the mass of the subhalos vs the number of LLSs."""
    #Mass
    subhalo_mass, coordinates = import_hdf5_subfind_folder(ProjectFiles.z3Subfind, part_type='Subhalo', hdf_paths=['StarFormationRate','CentreOfPotential'])
    subhalo_x = coordinates[:,0]
    subhalo_y = coordinates[:,1]
    subhalo_z = coordinates[:,2]


    # Create subhalo LLSs
    # subhalo  [ 1 ,  2 ,  3 , ..]
    # LLS      [[5],[12],[45], ..]

    grid = np.load(ProjectFiles.z3gridGalHIHalo)#np.load(ProjectFiles.z3gridGalHIHalfMassRad4)

    # Create and save -- OR -- load
    #plt.figure()
    #plt.imshow(np.log10(grid), origin='lower')
    #plt.show()

    groupsSubhalo, subhalo_masses = FoF2wGroupNums(grid, subhalo_x, subhalo_y, subhalo_mass)
    np.save("Galaxies/subhalogroups", groupsSubhalo)
    #np.save("Galaxies/subhalomasses", subhalo_masses)
    np.save("Galaxies/subhalostarformationrates", subhalo_masses)
    # load
    groups = np.load("Galaxies/subhalogroups.npy")
    #subhalo_masses = np.load("Galaxies/subhalomasses.npy")
    subhalo_masses = np.load("Galaxies/subhalostarformationrates.npy")

    # Convert the from coordinates to number
    nGroups = np.zeros((len(groups)))
    for i in xrange(0, len(groups), 1):
        nGroups[i] = len(groups[i])

    # Convert units
    subhalo_masses = np.array(subhalo_masses) * Snapshot.U_M/ Snapshot.HubbleParam  # cgs
    subhalo_masses /= Constant.M0                                                   # sun masses

    # Plot mean and mode
    mean_mass = np.sum(np.array(nGroups)*subhalo_masses)/np.sum(nGroups)
    mode_mass = subhalo_masses[np.argmax(nGroups)]

    plt.figure()
    plt.semilogx()

    #plt.semilogy()
    #plt.tight_layout()
    plt.scatter(subhalo_masses, nGroups)
    plt.xlabel(r" $ \rm The \ total \ projected \ metallicity} \ \alpha^{-1} $", labelpad=0)
    plt.ylabel(r" $ \rm The \ total \ projected \ metallicity} $", labelpad=0)

    #plt.xlabel(r"$ \rm \textbf{Mass of subhalo (M0)} $")
    #plt.ylabel(r"$ \rm \textbf{Number of LLSs in a subhalo} $")
    plt.axvline(x=mean_mass, color='r', linestyle='--')
    plt.axvline(x=mode_mass, color='g', linestyle='--')
    plt.ylim((0, max(nGroups)))
    plt.xlim((min(subhalo_masses), max(subhalo_masses)))
    plt.savefig("TEST.pdf", dpi=600, bbox_inches='tight')
    plt.show()

    # Plot above but with total LLSs per bin

    # Bin data into subhalo-masses (using total in each bin not mean)

    rough_diff = np.abs(np.log10(subhalo_masses[0]) - np.log10(subhalo_masses[1]))/6
    subhalo_masses, nGroups, mass_err = bin_data_x_space_total(np.log10(subhalo_masses), nGroups, bin_width=rough_diff)

    # Convert to log space
    mean_mass = np.log10(mean_mass)
    #mode_mass = np.log10(mode_mass)

    fig = plt.figure()
    #plt.semilogx()
    #plt.semilogy()
    mass_widths = 0.8*(subhalo_masses[1]-subhalo_masses[0])
    #plt.scatter(subhalo_masses, nGroups)
    #plt.errorbar(subhalo_masses, nGroups, xerr=mass_err, color='b', marker='o', linestyle='none')
    plt.bar(subhalo_masses, nGroups, width=mass_widths, align='center')
    plt.xlabel(r"$ \rm \log_{10}(Mass \ of \ subhalo/M_{0}) $", labelpad=3)
    plt.ylabel(r"$ \rm Total \ number \ of \ LLSs $", labelpad=2)
    plt.axvline(x=mean_mass, color='r', linestyle='--')
    #plt.axvline(x=mode_mass, color='g', linestyle='--')
    plt.ylim((0, max(nGroups)+80))
    padx = 0.05
    pady = 30
    plt.text(x=mean_mass+padx,y=plt.ylim()[1]-pady, s="Mean", verticalalignment='top', horizontalalignment='left', color='r')

    fig.tight_layout()
    plt.savefig("NLLSvsMassSubhalo.pdf", dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.show()


def subhaloVsNumberLLS():
    """ 1. Plot the number of LLSs against subhalo mass, with 3 curves: Total LLSs, CGM LLSs, <Half-mass radius .
        2. Plot the number of LLSs against the subhalo star formation rate, with "------------------------------------".
        """
    create_curves1 = 0
    create_curves2 = 1

    # 1.1. Create curves
    if create_curves1:
        subhalo_mass, coordinates = import_hdf5_subfind_folder(ProjectFiles.z3Subfind, part_type='Subhalo', hdf_paths=['Mass','CentreOfPotential'])
        subhalo_x = coordinates[:,0]
        subhalo_y = coordinates[:,1]
        subhalo_z = coordinates[:,2]

        # 1.1.1. Total LLSs vs subhalo mass
        grid_total = np.load(ProjectFiles.z3gridHI)
        grid_total = mask_all_but_LLS(grid_total)

        groups_total, subhalo_masses_total = FoF2wGroupNums(grid_total, subhalo_x, subhalo_y, subhalo_mass)
        np.save("Galaxies/groups_total", groups_total)
        np.save("Galaxies/subhalo_masses_total", subhalo_masses_total)

        # 1.1.2. CGM LLSs vs subhalo mass
        grid_cgm = np.load(ProjectFiles.z3gridGalHIHalo)
        grid_cgm = mask_all_but_LLS(grid_cgm)

        groups_cgm, subhalo_masses_cgm = FoF2wGroupNums(grid_cgm, subhalo_x, subhalo_y, subhalo_mass)
        np.save("Galaxies/groups_cgm", groups_cgm)
        np.save("Galaxies/subhalo_masses_cgm", subhalo_masses_cgm)

        # 1.1.3. < Half mass radius (hmr) of LLSs vs subhalo mass
        grid_hmr = np.load(ProjectFiles.z3gridGalHIHalfMassRad4)
        grid_hmr = mask_all_but_LLS(grid_hmr)

        groups_hmr, subhalo_masses_hmr = FoF2wGroupNums(grid_hmr, subhalo_x, subhalo_y, subhalo_mass)
        np.save("Galaxies/groups_hmr", groups_hmr)
        np.save("Galaxies/subhalo_masses_hmr", subhalo_masses_hmr)

    # 1.2. Get arrays ready for plotting
    # 1.2.1. Load arrays
    groups_tot          = np.load("Galaxies/groups_total.npy")
    subhalo_masses_tot  = np.load("Galaxies/subhalo_masses_total.npy")
    groups_cgm          = np.load("Galaxies/groups_cgm.npy")
    subhalo_masses_cgm  = np.load("Galaxies/subhalo_masses_cgm.npy")
    groups_hmr          = np.load("Galaxies/groups_hmr.npy")
    subhalo_masses_hmr  = np.load("Galaxies/subhalo_masses_hmr.npy")

    # 1.2.2. Convert arrays from list of coordinate tuples to single arrays of number
    n_groups_tot = np.zeros((len(groups_tot)))
    n_groups_cgm = np.zeros((len(groups_cgm)))
    n_groups_hmr = np.zeros((len(groups_hmr)))

    print len(groups_tot), len(groups_cgm), len(groups_hmr)
    for i in xrange(0, len(groups_tot), 1):         # Assumes all have the same length
        n_groups_tot[i] = len(groups_tot[i])
    for i in xrange(0, len(groups_cgm), 1):
        n_groups_cgm[i] = len(groups_cgm[i])
    for i in xrange(0, len(groups_hmr), 1):
        n_groups_hmr[i] = len(groups_hmr[i])


    # All subhalo masses should be the same
    #assert subhalo_masses_cgm == subhalo_masses_hmr == subhalo_masses_tot
    #subhalo_masses = subhalo_masses_tot

    # Convert units
    subhalo_masses_tot  = np.array(subhalo_masses_tot) * Snapshot.U_M/ Snapshot.HubbleParam  # cgs
    subhalo_masses_tot /= Constant.M0                                                        # sun masses
    subhalo_masses_cgm  = np.array(subhalo_masses_cgm) * Snapshot.U_M/ Snapshot.HubbleParam  # cgs
    subhalo_masses_cgm /= Constant.M0
    subhalo_masses_hmr  = np.array(subhalo_masses_hmr) * Snapshot.U_M/ Snapshot.HubbleParam  # cgs
    subhalo_masses_hmr /= Constant.M0

    # Plot mean and mode
    #mean_mass = np.sum(np.array(nGroups)*subhalo_masses)/np.sum(nGroups)
    #mode_mass = subhalo_masses[np.argmax(nGroups)]

    #plt.figure()
    #plt.semilogx()
    #plt.scatter(subhalo_masses, nGroups)
    #plt.xlabel(r" $ \rm The \ total \ projected \ metallicity} \ \alpha^{-1} $", labelpad=0)
    #plt.ylabel(r" $ \rm The \ total \ projected \ metallicity} $", labelpad=0)

    #plt.xlabel(r"$ \rm \textbf{Mass of subhalo (M0)} $")
    #plt.ylabel(r"$ \rm \textbf{Number of LLSs in a subhalo} $")
    #plt.axvline(x=mean_mass, color='r', linestyle='--')
    #plt.axvline(x=mode_mass, color='g', linestyle='--')
    #plt.ylim((0, max(nGroups)))
    #plt.xlim((min(subhalo_masses), max(subhalo_masses)))
    #plt.savefig("TEST.pdf", dpi=600, bbox_inches='tight')
    #plt.show()

    # Plot above but with total LLSs per bin
    # Bin data into subhalo-masses (using total in each bin not mean)
    rough_diff = np.abs(np.log10(subhalo_masses_tot[0]) - np.log10(subhalo_masses_tot[1]))/20
    subhalo_masses_tot, n_groups_tot, mass_err_tot = bin_data_x_space_total(np.log10(subhalo_masses_tot), n_groups_tot, bin_width=rough_diff)
    subhalo_masses_cgm, n_groups_cgm, mass_err_cgm = bin_data_x_space_total(np.log10(subhalo_masses_cgm), n_groups_cgm, bin_width=rough_diff)
    subhalo_masses_hmr, n_groups_hmr, mass_err_hmr = bin_data_x_space_total(np.log10(subhalo_masses_hmr), n_groups_hmr, bin_width=rough_diff)

    # Convert to log space
    #mean_mass = np.log10(mean_mass)
    fig = plt.figure()

    # Convert to log space
    plt.semilogy()


    mass_widths = 0.03#1000.0*rough_diff
    #plt.scatter(subhalo_masses, nGroups)
    #plt.errorbar(subhalo_masses, nGroups, xerr=mass_err, color='b', marker='o', linestyle='none')
    #plt.errorbar(subhalo_masses_tot, n_groups_tot, label="Total", color='b', marker='o', linestyle='none')
    #plt.errorbar(subhalo_masses_cgm, n_groups_cgm,  label="CGM", color='g', marker='o', linestyle='none')
    #plt.errorbar(subhalo_masses_hmr, n_groups_hmr, label="Core", color='r', marker='o', linestyle='none')

    plt.bar(subhalo_masses_tot, n_groups_tot, width=mass_widths, align='center', label="Total", color='b', edgecolor='none', log=True)
    plt.bar(subhalo_masses_cgm, n_groups_cgm, width=mass_widths, align='center', label="CGM", color='g', edgecolor='none', log=True)
    plt.bar(subhalo_masses_hmr, n_groups_hmr, width=mass_widths, align='center', label="Core", color='r', edgecolor='none', log=True)

    plt.xlabel(r"$ \rm \log_{10}(Mass \ of \ subhalo/M_{0}) $", labelpad=3)
    plt.ylabel(r"$ \rm Total \ number \ of \ LLSs $", labelpad=2)
    #plt.axvline(x=mean_mass, color='r', linestyle='--')
    #plt.axvline(x=mode_mass, color='g', linestyle='--')
    plt.ylim((0, max(n_groups_tot)+80))
    padx = 0.05
    pady = 30
    #plt.text(x=mean_mass+padx,y=plt.ylim()[1]-pady, s="Mean", verticalalignment='top', horizontalalignment='left', color='r')

    fig.tight_layout()
    plt.legend(frameon=False)
    plt.savefig("NLLSvsMassSubhaloTotCGMCore.pdf", dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.show()


def get_error_on_number_of_LLSs():
    grid1 = np.load(ProjectFiles.z3gridHIhalf1)
    grid2 = np.load(ProjectFiles.z3gridHIhalf2)
    grid1 = mask_all_but_LLS(grid1)
    grid2 = mask_all_but_LLS(grid2)

    N1 = len(grid1.nonzero()[0])
    N2 = len(grid2.nonzero()[0])

    print "Number of LLSs in grid1: ", N1
    print "Number of LLSs in grid2: ", N2
    print "Error = difference = : ", (N2 - N1)
    print "% error on N2: ", np.abs(N2 - N1)/N2
    print "% error on N1: ", np.abs(N2 - N1)/N1


def star_formation_vs_mass():
    subhalo_mass, star_formation = import_hdf5_subfind_folder(ProjectFiles.z3Subfind, part_type='Subhalo', hdf_paths=['Mass','StarFormationRate'])
    plt.figure()
    plt.plot(subhalo_mass, star_formation)
    plt.show()


def plot_individual_correlation_on_axis(ax, all_x, all_y):

    plot_all = 1
    plot_means = 1
    plot_total_mean = 1
    plot_mean_of_all_LLS = 1
    plot_sum_of_means = 0

    colours = cm.rainbow(np.linspace(0, 1, len(all_y)+1))
    lines = ['--',':', '-', '-.']

    if plot_all:
        maxima = []
        for i in xrange(0, len(all_x),1):
            ax.plot(all_x[i], all_y[i], color=colours[i], alpha = 0.2)
            if all_y[i] != []:
                maxima.append( max(all_y[i]) )
                #ax.plot(all_x[i], all_y[i]/max(all_y[i]), color=colours[i])
            else:
                pass
                #ax.plot(all_x[i], all_y[i], color=colours[i])

    if plot_means:
        for i in xrange(0, len(all_y), 1):
            #print all_x[i]
            x, y, y_err = bin_data_fixed_bins(all_x[i], all_y[i], 1, range=[0, 200])
            print "Number of mean data points", len(x)
            ax.plot(x, y, color=colours[i], linewidth=1.5, linestyle=lines[2])

    # Plot the mean of all individual LLSs
    if plot_total_mean:
        mean_y = np.zeros((199))
        for i in xrange(0, len(all_y), 1):
            x, y, y_err = bin_data_fixed_bins(all_x[i], all_y[i], 1, range=[0, 200])
            for j in xrange(len(y)):
                if np.isnan(y[j]):
                    pass
                else:
                    mean_y[j] += y[j]
                    #print mean_y
        mean_y /= len(all_y)
        print x, mean_y
        ax.plot(x, mean_y, color='black', linewidth=2)

    #
    if plot_sum_of_means:
        mean_y = np.zeros((199))
        for i in xrange(0, len(all_y), 1):
            x, y, y_err = bin_data_fixed_bins(all_x[i], all_y[i], 1, range=[0, 200])
            for j in xrange(len(y)):
                if np.isnan(y[j]):
                    pass
                else:
                    mean_y[j] += y[j]
                    #print mean_y
        #mean_y /= len(all_y)
        print x, mean_y
        ax.plot(x, mean_y, color='black', linewidth=2)

    # Plot the mean correlation of all LLSs
    if plot_mean_of_all_LLS:
        grid = np.load(ProjectFiles.z3gridHI)
        grid = mask_all_but_LLS(grid)
        x, y = auto_correlation_two_point_fourier(grid)
        x *= cell_to_pkpc
        ax.plot(x, y, color = 'blue', alpha=0.3)
        x2,y2,x_err, y_err = bin_data_x_space(x, y, 1, np.sqrt(2)/2)#bin_data_fixed_bins(x,y,bin_width=1, range=[0, 200])
        ax.plot(x2,y2,color='blue', linewidth=3, linestyle='--')
    return ax


def plot_individual_correlations():
    groups_total = np.load("Galaxies/groups_total.npy")

    create = 0              # create again and override if new grid
    if create:
        all_x = []
        all_y = []
        for i in xrange(0, len(groups_total), 1):
            group = groups_total[i]
            idx_i, idx_j = zip(*group)
            idx_i, idx_j = np.array(idx_i), np.array(idx_j)

            #centre on LLSs
            idx_i -= int(np.mean(idx_i))
            idx_j -= int(np.mean(idx_j))

            small_grid = np.zeros((max(idx_i)-min(idx_i)+1, 1+max(idx_j)-min(idx_j)))
            small_grid[idx_i, idx_j] = 1
            x, y = auto_correlation_two_point_fourier(small_grid)
            all_x.append(x)
            all_y.append(y)

        np.save("Galaxies/all_x.npy", all_x)
        np.save("Galaxies/all_y.npy", all_y)

    else:
        all_x = np.load("Galaxies/all_x.npy")
        all_y = np.load("Galaxies/all_y.npy")
    print "Number of grouops", len(all_y)

    # Convert units
    for i in xrange(len(all_x)):
        all_x[i] *= cell_to_pkpc

    maximum = max(convert_list_of_lists_to_list(all_y))

    # PLOTTING
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #plt.semilogx(nonposx='clip')
    #plt.semilogy(nonposy='clip')
    #plt.loglog()
    #colours = ['b', 'r', 'g','black','grey','y','m','c']

    # 1. Add plot to axes
    ax = plot_individual_correlation_on_axis(ax, all_x, all_y)

    # 1.1. Add cosmetics to axes
    ax.set_xlabel(r"$ \rm |\Delta| $"+"$ \ (pkpc) $", labelpad=3)
    ax.set_ylabel(r"$ \rm \xi (\Delta) $", labelpad=2)
    ax.axhline(y=0, color='black', linestyle=':')
    ax.set_ylim((-1, maximum))
    ax.set_xlim((0, 80))
    ax.xaxis.set_major_formatter(ScalarFormatter())             # Set scalar numbers on tick labels, not 10^X
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_on()                                          # Increase size of ticks
    ax.tick_params('both', length=6, width=1, which='major')
    ax.tick_params('both', length=3, width=1, which='minor')

    # 2. Add inset
    axins = inset_axes(ax,
               width="60%",     # width = 30% of parent_bbox
               height="60%",    # height : 1 inch
               loc=1,
               borderpad=0.8)#,
               #bbox_transform=ax.figure.transFigure)

    # 2.1. Plot everything on inset again
    axins = plot_individual_correlation_on_axis(axins, all_x, all_y)

    # 2.2. Cosmetics for inset
    axins.set_xlim((0, 50))
    axins.set_ylim((-1, 8))
    axins.xaxis.set_major_formatter(ScalarFormatter())             # Set scalar numbers on tick labels, not 10^X
    axins.yaxis.set_major_formatter(ScalarFormatter())
    axins.minorticks_on()                                          # Increase size of ticks
    axins.tick_params('both', length=2, width=0.5, which='major')
    axins.tick_params('both', length=1, width=0.5, which='minor')
    axins.axhline(y=0, color='black', linestyle=':')

    # 3. Save and display figure
    plt.savefig("Correlation Function/Report/CorrelationOfIndividualLLS4"+".pdf", dpi=1000, bbox_inches='tight', pad_inches=0.04)
    plt.savefig("Correlation Function/Report/CorrelationOfIndividualLLS4"+".png", dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.savefig("Correlation Function/Report/CorrelationOfIndividualLLS4"+".jpg", dpi=300, bbox_inches='tight', pad_inches=0.04)

    plt.show()


def plot_difference_vs_size():
    # 1.2.1. Load arrays
    groups_tot          = np.load("Galaxies/groups_total.npy")          # LLS groups for entire grid
    subhalo_masses_tot  = np.load("Galaxies/subhalo_masses_total.npy")
    groups_cgm          = np.load("Galaxies/groups_cgm.npy")
    subhalo_masses_cgm  = np.load("Galaxies/subhalo_masses_cgm.npy")

    # 1.2.2. Convert arrays from list of coordinate tuples to single arrays of number
    n_groups_tot = np.zeros((len(groups_tot)))
    n_groups_cgm = np.zeros((len(groups_cgm)))

    #print len(groups_tot), len(groups_cgm), len(groups_hmr)
    for i in xrange(0, len(groups_tot), 1):         # Assumes all have the same length
        n_groups_tot[i] = len(groups_tot[i])
    for i in xrange(0, len(groups_cgm), 1):
        n_groups_cgm[i] = len(groups_cgm[i])



    # Bin data into subhalo-masses (using total in each bin not mean)
    rough_diff = np.abs(np.log10(subhalo_masses_tot[0]) - np.log10(subhalo_masses_tot[1]))/200
    range = [np.log10(min(subhalo_masses_tot)), np.log10(max(subhalo_masses_tot))]
    subhalo_masses_tot, n_groups_tot, mass_err_tot = bin_data_fixed_bins(np.log10(subhalo_masses_tot), n_groups_tot, range=range, bin_width=rough_diff, total=True)
    subhalo_masses_cgm, n_groups_cgm, mass_err_cgm = bin_data_fixed_bins(np.log10(subhalo_masses_cgm), n_groups_cgm, range=range, bin_width=rough_diff, total=True)

    subhalo_masses_tot = 10**subhalo_masses_tot
    subhalo_masses_cgm = 10**subhalo_masses_cgm

    plt.figure()
    plt.scatter(subhalo_masses_cgm, n_groups_tot-(n_groups_cgm))
    #plt.plot(subhalo_masses_tot, n_groups_tot)
    #plt.plot(subhalo_masses_cgm, n_groups_cgm)

    plt.show()


# Import data ======================================================================================================== #

# Use fake data for debugging and testing
if use_fake == 1:
    fake_grid = ma.zeros((4,4))
    fake_grid[0:3,0:3] = 1e18
    idxs    = [[1,0],[1,1],[2,1]]
    fake_grid[zip(*idxs)] = ma.masked
    grid    = np.flipud(fake_grid)
    groups  = FoF2(grid)

# Import actual column density grid and if load_grps == 1 import groups from saved file
else:
    grid = np.load(grid_folder + grid_name)
    if load_grps == 1:
        groups = np.load(save_folder + "3NGrids3RMin128RMax4096NDGrid3Groups1200.npy")
    else:
        groups = FoF2(grid)

# Save the result for use again at a later time
if save_grps == 1:
    np.save(save_folder + base_save_name + save_name, groups)

    # Save vertically flipped grid:                     Flip i values for plotting: array's index starts at the top left, but plotting index starts at bottom left.
    groups = flipud_groups(groups, grid.shape)
    np.save(save_folder + base_save_name + "VFlipped" + save_name , groups)

# Define grid-specific parameters
NcellsX      = grid.shape[0]                            # Number
cell_to_pkpc = Snapshot.Boxsizepkpc/NcellsX             # pkpc
cell_to_ckpc = Snapshot.Boxsizeckpc/NcellsX
cell_to_pMpc = Snapshot.BoxsizepMpc/NcellsX             # pMpc
cell_to_cMpc = Snapshot.BoxsizecMpc/NcellsX             # cMpc
cell_to_cMpch= Snapshot.BoxsizecMpch/NcellsX
print "Each grid cell correpsonds to ", cell_to_pkpc, " physical kpc (pkpc)"

grid = ma.masked_where(np.log10(grid) < 17.2, grid)                         # Isolate LLs Pixels
grid = ma.masked_where(np.log10(grid) > 20.3, grid)

if PLOT_GRID == 1:
    plt.figure()                                                                # Plot Colour map of LLs pixels
    plt.imshow(np.log10(grid), cmap = 'YlOrRd', origin = 'lower', interpolation='none', aspect = 'equal')#, extent = (xRange[0]/cLen, xRange[1]/cLen, yRange[0]/cLen, yRange[1]/cLen))
    plt.colorbar()
    plt.xlabel("X-axis Cell Number " )
    plt.ylabel("Y-axis Cell Number ")
    #plt.title(r"$\rm  log_{10} (N_{HI}) \ [HI \ atoms/cm^{2}]$")
    #plt.gca().set_aspect(1./plt.gca().get_data_ratio())
    #plt.savefig(save_folder + base_save_name + "LLSColourMapNoGroups" + str(dpi) + ".png", dpi = dpi)
    plt.show()

# Plot Colour map of LLs pixels
if PLOT_GROUPS_COLOUR_MAP == 1:
    # Convert groups (list of lists of indexes) to a 2D numpy array
    grid = writeGroupsToArray(groups, grid)

    plt.figure()
    plt.imshow(np.log10(grid), cmap = 'YlOrRd', origin = 'lower', interpolation='none', aspect = 'equal')#, extent = (xRange[0]/cLen, xRange[1]/cLen, yRange[0]/cLen, yRange[1]/cLen))
    plt.colorbar()
    plt.xlabel("X-axis Cell Number " )
    plt.ylabel("Y-axis Cell Number ")
    plt.gca().set_aspect(1./plt.gca().get_data_ratio())
    plt.show()

# Plot all groups in a scatter plot
if PLOT_GROUPS_SCATTER == 1:
    colour         = ['m', 'k', 'c', 'b', 'r', 'y' , 'g']
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(len(groups)):
        x, y = zip(*groups[i])
        ax.scatter(y, x, marker='s', color = colour[i % 7], edgecolors='none', s = 0.2 )
    plt.xlim((0,grid.shape[0]))
    plt.ylim((0,grid.shape[1]))
    plt.xlabel("X-axis Cell Number" )
    plt.ylabel("Y-axis Cell Number" )
    plt.gca().set_aspect(1./plt.gca().get_data_ratio())
    #plt.savefig(save_folder + base_save_name + "group_colouring" + str(dpi) + ".png")
    plt.show()

# Plot surface area/total area vs number of LLS with said surface area
if PLOT_HISTOGRAM_SURFACE_AREA == 1:
    pixel_area = cell_to_pkpc**2
    surface_area = np.zeros(len(groups))
    for i in xrange(len(groups)):
        surface_area[i] = len(groups[i]) * pixel_area
    center, hist = histogram(surface_area)

    # plot the histogram
    plt.figure()
    plt.scatter(np.log10(center), np.log10(hist))
    plt.xlabel(r"$ \rm \log_{10} \ surface \ area  \ \   (pkpc^{2}) $")
    plt.ylabel(r"$ \rm \log_{10} \ LLS \ with \ area \ <= \ size  $")
    plt.show()


# Plots cumulative fraction of pixels in LLS smaller than a given size
if FRACTION_AREA_IN_LLS == 1:
    # Select units to use
    A_units = "pkpc"
    A_units_graph = r"$ \rm pkpc^{2}$"            # Units of area
    pixel_area = cell_to_pkpc**2

    surface_area = np.zeros(len(groups), dtype=int)
    for i in xrange(len(groups)):
        surface_area[i] = len(groups[i]) #* pixel_area
    frac_surface_area   = surface_area / np.sum(surface_area)

    cum_sum = []
    print "Mean surface area: ", np.mean(surface_area*pixel_area), A_units
    print "Std error in surface area: ",np.sqrt( np.sum( (surface_area*pixel_area - np.mean(surface_area*pixel_area) ) **2 ) / (len(surface_area*pixel_area) - 1) ) / np.sqrt(len(surface_area*pixel_area)), A_units
    fake_surface_area = np.arange(1, max(surface_area) + 1,1)
    for i in xrange(1, max(surface_area) + 1,1):
        cum_sum.append(np.sum(surface_area[np.where(surface_area == i)]))

    cum_sum = np.cumsum(cum_sum)
    cum_sum = cum_sum/max(cum_sum)
    # Convert units from cells to pixels
    fake_surface_area *= pixel_area

    print len(surface_area), len(cum_sum)

    # Error
    #x_error = x*np.sqrt(2)*len_error/len      # Z = A*B : a_Z = Z*sqrt( (a_A/A)**2 + (a_B/B)**2 )
    fake_surface_area_error = fake_surface_area

    # Plot histogram
    center, hist = histogram(surface_area*pixel_area, 10000)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.semilogx()
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    # Eliminate non-zeros
    nonzeros = np.where(hist != 0)
    center = center[nonzeros]
    hist = hist[nonzeros]


    ax.scatter(center, hist)

    #plt.hist(surface_area, bins = 1000)
    #ax.bar(center, hist, width=1.0)#, log=True)
    #ax.bar(np.log10(center), hist,width=0.01)#, log=True)
    ax.set_ylim((0, max(hist)+4))

    # Plot lines
    mean_x = np.mean(surface_area*pixel_area)
    plt.axvline(x=mean_x, color='r', linestyle='--')
    #plt.text(x=mean_x,y=plt.ylim()[1],s="Mean", verticalalignment='bottom', horizontalalignment='left', color='r')
    plt.text(x=1.1*mean_x,y=0.96*plt.ylim()[1],s="Mean", verticalalignment='top', horizontalalignment='left', color='r')

    total = np.sum(hist)

    # Find fraction of LLSs in region 0.01 to 100 pkpc^2 (galaxy sizes)
    big_gal_idx = find_nearest(center, 100)
    print "Fraction of LLSs in galaxy range: ", np.sum(hist[0:big_gal_idx])/total

    # Plot quartitles
    #
    #cum_freq = np.cumsum(hist)

    #Quartile1 = find_nearest(cum_freq, 0.25*total)
    #Quartile2 = find_nearest(cum_freq, 0.50*total)
    #Quartile3 = find_nearest(cum_freq, 0.75*total)

    #plt.axvline(center[Quartile1], color='c', linestyle='--')
    #plt.axvline(center[Quartile2], color='c', linestyle='--')
    #plt.axvline(center[Quartile3], color='c', linestyle='--')


    # Plot mode
    mode_x = center[np.argmax(hist)]
    plt.axvline(x=mode_x, color='black', linestyle='--')
    #plt.text(x=mode_x,y=plt.ylim()[1],s="Mode", verticalalignment='bottom', horizontalalignment='right')
    plt.text(x=0.75*mode_x,y=0.96*plt.ylim()[1],s="Mode", verticalalignment='top', horizontalalignment='right')

    ax.minorticks_on()                                          # Increase size of ticks
    ax.tick_params('both', length=4, width=1, which='major')
    ax.tick_params('both', length=2, width=1, which='minor')

    # Axis labels
    ax.set_xlabel(r"$ \rm Surface \ area  \ \   ($"+A_units_graph+"$) $", labelpad=2)
    ax.set_ylabel(r"$ \rm Number \ of \ LLSs \ with \ given \ size $", labelpad=3)

    print "Modal surface area: ", mode_x, A_units
    print "Mean surface area : ", mean_x, A_units

    plt.savefig("LLSArea/LLSAreaHistogram2"+A_units+".pdf", dpi = 600, bbox_inches='tight', pad_inches=0.04)
    plt.show()

    # Plot fractional surface area
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.semilogx()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    ax.minorticks_on()
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.tick_params('both', length=10, width=1, which='major')
    ax.tick_params('both', length=5 , width=1, which='minor')
    plt.plot(fake_surface_area, cum_sum)
    #plt.axhline(y=0.5,xmin=0, xmax=0.53, color='r', linestyle='--')
    #plt.axvline(x=np.mean(surface_area*pixel_area), ymin=0, ymax=0.5, color='r')
    #ax.errorbar(fake_surface_area, cum_sum, xerr=fake_surface_area_error)
    ax.set_xlabel(r"$ \rm Surface \ area  \ \   (pkpc^{2}) $")
    ax.set_ylabel(r"$ \rm Fractional \ area \ in \ LLS \ of \ a \ given \ size $")
    ax.set_xlim((0, max(fake_surface_area)))
    ax.set_ylim((0, max(cum_sum)))

    # Uncomment for an axis insert
    #axins = inset_axes(ax,
    #           width="60%",     # width = 30% of parent_bbox
    #           height="60%",    # height : 1 inch
    #           loc=4,
    #           borderpad=1.6)

    #axins.scatter(fake_surface_area, cum_sum)
    #axins.set_xlim((0, 800))
    #axins.set_ylim((0, 0.8))

    plt.savefig("LLSArea/CumulativeFractionPixelsLEQSize.png", dpi = 600, bbox_inches='tight')
    plt.show()

    # Plots number of pixels in LLS greater than a given size
    fake_LLS_sizes = np.arange(1, max(surface_area), 1)
    NpixGRE = []
    for i in fake_LLS_sizes:
        NpixGRE.append(np.sum(surface_area[np.where(surface_area > i)]))

    fake_LLS_sizes *= cell_to_pkpc

    plt.figure()
    plt.loglog()
    plt.scatter(fake_LLS_sizes, NpixGRE)


    plt.xlabel(r"$ \rm Surface \ area  \ \   (pkpc^{2}) $")
    plt.ylabel(r"$ \rm Total \ area \ in \ LLS \ with \ area \ > \ size  $")
    plt.xlim((0, max(fake_LLS_sizes)))
    plt.ylim((0, max(NpixGRE)))
    #plt.savefig("LLSArea/NumberPixelsGSize.png", dpi = 600, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #plot_difference_vs_size()
    plot_individual_correlations()
    #get_error_on_number_of_LLSs()
    #star_formation_vs_mass()
    #subhaloVsNumberLLS()
    #mass_subhaloVsNumberLLS()

    #grid1 = np.load(ProjectFiles.z3gridHI)
    #grid2 = np.load(ProjectFiles.z3gridAllSubhalos)
    
    #grid1 = mask_all_but_DLAs(grid1)
    
    #print "LLS cover ", 100*len(grid1.nonzero()[0])/(grid1.shape[0]*grid1.shape[1]), "%"
    
    #plt.figure()
    #plt.imshow(np.log10(grid2), origin='lower')
    #plt.show()
    
    #overlap(grid1, grid2)
    #plot_maximal_extent(groups)