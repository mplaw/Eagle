""" Import subfind files (currently they are used for the virial radii and half mass radii of the galaxies) """
from __future__ import division
from PhysicalConstants import *
from SnapshotParameters import *
from MiscellaneousFunctions import *
import ProjectFiles


__author__ = 'Matt'


# Note not all the group files have the same tabs in - some of the latter ones don't have any subhalo information.
# Hence the try and except clause in import_hdf5_subfind_file

# Import subfind files =============================================================================================== #


def import_hdf5_subfind_file(file_path='.', part_type=0, hdf_paths=['Group_R_Mean200', 'GroupCentreOfPotential']):
    """ Imports information from a single HDF5 file.
    :param file_path: A string containing the complete HDF5-file location, e.g. 'C:/Users/Matt/Documents/Uni Work 4/Project/Eagle Simulations/snap_028_z000p000'
    :param part_type: The type of particles to read information about: 0 gas, 1 dark matter, 4 stars, 5 black holes (2 and 3 are non-existent)
    :param hdf_paths: A list of strings, where each string is a name of a data folder in the hdf5 file.
    :return: A list of arrays, where each array is the data in the folder specified by the corresponding hdf_paths string. """
    f = h5py.File(file_path, 'r')
    data = []
    for i in hdf_paths:
        try:
            data.append(np.array(f[str(part_type) + '/' + i]))
        except:
            pass
            #print "No", str(part_type) + '/' + i, "found for", file_path
    f.close()
    return data


def import_hdf5_subfind_folder(folder_path='.', part_type=0, hdf_paths=['HalfMassRad', 'CentreOfPotential']):
    """ Import information from all HDF5 files in a folder
    :param folder_path: A string containing the folder with all the HDF5-files in, e.g. 'C:/Users/Matt/Documents/Uni Work 4/Project/Eagle Simulations'
    :param part_type: The type of particles to read information about: 0 gas, 1 dark matter, 4 stars, 5 black holes (2 and 3 are non-existent)
    :param hdf_paths: A list of strings, where each string is a name of a data folder in the hdf5 file.
    :return: A list of arrays, where each array is the data in the folder specified by the corresponding hdfPaths string.
    """
    dirs = os.listdir(folder_path)
    data = import_hdf5_subfind_file(folder_path + '/' + dirs[0], part_type=part_type, hdf_paths=hdf_paths)
    for i, fileName in enumerate(dirs[1:-1]):
        new_data = import_hdf5_subfind_file(folder_path + '/' + fileName, part_type=part_type, hdf_paths=hdf_paths)
        for j in range(len(new_data)):
            data[j] = np.concatenate((data[j], new_data[j]))
    return data


class SUBFIND:
    # Halo properties (FOF)
    halo_centre_x = None
    halo_centre_y = None
    halo_centre_z = None
    halo_virial_rad = None

    # Subhalo properties (== Galaxies) (SUBFIND)
    subhalo_centre_x = None
    subhalo_centre_y = None
    subhalo_centre_z = None
    subhalo_half_mass_rad = None
    subhalo_group_number = None
    subhalo_subgroup_number = None


    # Miscellaneous properties
    units = 'cMpch'                         # The units of the numbers

    def __init__(self, folder_path):
        print "Initialising"
        self.import_subfind(folder_path)


    def import_subfind(self, folder_path):
        self.subhalo_half_mass_rad, subcrds, self.subhalo_group_number, self.subhalo_subgroup_number = import_hdf5_subfind_folder(folder_path, 'Subhalo', hdf_paths=['HalfMassRad', 'CentreOfPotential', 'GroupNumber', 'SubGroupNumber'])
        self.subhalo_centre_x = subcrds[:, 0]
        self.subhalo_centre_y = subcrds[:, 1]
        self.subhalo_centre_z = subcrds[:, 2]
        subcrds = None
        del subcrds

        self.halo_virial_rad, crds = import_hdf5_subfind_folder(folder_path, 'FOF', hdf_paths=['Group_R_Mean200', 'GroupCentreOfPotential'])
        self.halo_centre_x = crds[:, 0]
        self.halo_centre_y = crds[:, 1]
        self.halo_centre_z = crds[:, 2]
        crds = None
        del crds
        #print np.max(self.halo_centre_x), np.max(self.subhalo_half_mass_rad), np.max(self.subhalo_half_mass_rad)

    def convert_units(self, unit_conversion):
        self.halo_centre_x *= unit_conversion
        self.halo_centre_y *= unit_conversion
        self.halo_centre_z *= unit_conversion
        self.halo_virial_rad *= unit_conversion
        self.subhalo_centre_x *= unit_conversion
        self.subhalo_centre_y *= unit_conversion
        self.subhalo_centre_z *= unit_conversion
        self.subhalo_half_mass_rad *= unit_conversion

    def convert_to_pkpc(self):
        if self.units == 'cMpch':
            self.convert_units(1000*Snapshot.ExpansionFactor/Snapshot.HubbleParam)
            self.units = 'pkpc'
        elif self.units == 'cMpc':
            self.convert_units(1000*Snapshot.ExpansionFactor)
            self.units = 'pkpc'
        elif self.units == 'pMpc':
            self.convert_units(1000)
            self.units = 'pkpc'
        elif self.units == 'pkpc':
            print "Already in pkpc."
        else:
            print "Unit's are not a standard type [cMpch, cMpc, pMpc]. Please convert back to a standard type before using " \
                  "this method."

    def convert_to_cMpch(self):
        if self.units == 'Mpc':
            self.convert_units(Snapshot.HubbleParam/Snapshot.ExpansionFactor)
            self.units = 'cMpch'
        elif self.units == 'pkpc':
            self.convert_units(Snapshot.HubbleParam/1000*Snapshot.ExpansionFactor)
            self.units = 'cMpch'
        elif self.units == 'cMpc':
            self.convert_units(Snapshot.HubbleParam)
            self.units = 'cMpch'
        elif self.units == 'pMpc':
            self.convert_units(Snapshot.HubbleParam/Snapshot.ExpansionFactor)
            self.units = 'cMpch'
        elif self.units == 'cMpch':
            print "Units are already in cMpch"
        else:
            print "Unit's are not a standard type [cMpch, cMpc, pMpc]. Please convert back to a standard type before using " \
                  "this method."


# Test code: Plots an overlay of the grid and the virial radius + subhalo radius
if __name__ == '__main__':

    small_fig_width_pt = 223.51561 + 18 #246.0  # Get this from LaTeX using \showthe\columnwidth
    big_fig_width_pt = 526.72534 + 25

    fig_width_pt = small_fig_width_pt
    inches_per_pt = 1.0/72.27               # Convert pt to inches
    golden_mean = 0.772 #5**0.5-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height =fig_width*golden_mean       # height in inches
    fig_size = [fig_width,fig_height]

    params = {'backend': 'ps',
            'mathtext.default': 'regular',
            'font.family'  : 'Times',
            'font.size'      : 11,
            'font.weight' : 400,#'bold',
            'axes.labelsize' : 11,
            'legend.fontsize': 11,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'text.usetex': False,
            'text.latex.unicode' : True,
            'figure.figsize': fig_size}
    plt.rcParams.update(params)         # Set custom font for graph labels

    full_grid = np.load(ProjectFiles.z3gridHI)
    subfind = SUBFIND(ProjectFiles.z3Subfind)

    fig = plt.figure()

    # Pick units for colour map and circles
    extent = (0, Snapshot.Boxsizepkpc, 0, Snapshot.Boxsizepkpc)
    subfind.convert_to_pkpc()

    # Plot colour map of the full grid
    plt.imshow(np.log10(full_grid), origin='lower', extent=extent, cmap=colour_map_split3(np.log10(full_grid)))
    cbar = plt.colorbar(pad=0)
    cbar.set_label(r"$ \rm log_{10} \ (N_{HI}/cm^{-2}) $", rotation=-90, labelpad=15)

    # Plot virial radius of FOF haloes
    for i in xrange(0, len(subfind.halo_centre_x)):
        circle = plt.Circle((subfind.halo_centre_x[i], subfind.halo_centre_y[i]), radius=subfind.halo_virial_rad[i],
                            facecolor="none", edgecolor='black', alpha=0.7, linestyle='dotted', linewidth=1)
        fig.gca().add_artist(circle)

    # Plot half-mass-radius of sub-haloes
    for j in xrange(0, len(subfind.subhalo_centre_x), 1):
        circle = plt.Circle((subfind.subhalo_centre_x[j], subfind.subhalo_centre_y[j]), radius=subfind.subhalo_half_mass_rad[:,0][j],
                            facecolor="none", edgecolor='black', linestyle='solid')
        fig.gca().add_artist(circle)

    plt.xlim((0, Snapshot.Boxsizepkpc))
    plt.ylim((0, Snapshot.Boxsizepkpc))
    plt.xlabel(r"$ \rm x \ (pkpc)$", labelpad=3)
    plt.ylabel(r"$ \rm y \ (pkpc)$", labelpad=2)
    plt.savefig("Galaxies/z3FullGridSubfindHalosDotted2"+".pdf", dpi=1000, bbox_inches='tight', pad_inches=0.04)
    plt.savefig("Galaxies/z3FullGridSubfindHalosDotted2"+".png", dpi=1000, bbox_inches='tight', pad_inches=0.04)

    # Close up of rightmost over density
    plt.xlim((2700, 3100))
    plt.ylim((700,  1100))

    plt.xticks([2700, 2800, 2900, 3000, 3100])
    plt.yticks([700, 800, 900, 1000, 1100])

    plt.savefig("Galaxies/z3FullGridSubfindHalosCloseupDotted2.pdf", dpi=1000, bbox_inches='tight', pad_inches=0.04)
    plt.savefig("Galaxies/z3FullGridSubfindHalosCloseupDotted2.png", dpi=1000, bbox_inches='tight', pad_inches=0.04)

    plt.show()