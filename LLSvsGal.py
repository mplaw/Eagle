""" Contains all the code for comparing LLSs to Galaxies """

from __future__ import division
from PhysicalConstants import *
from Subfind_file import *
from ProjectionFunctions import *
from MiscellaneousFunctions import *

__author__ = 'Matt'

subfind = SUBFIND(ProjectFiles.z3Subfind)
plt.rc('text', usetex=True)                                 # Set pyplot strings to use LaTeX formatting
plt.rc('font', family='Times', size=20)                     # Set custom font for graph labels

# Isolate gas in galaxies ==================================================================================================================================== #


def isolate_gas_galaxies_old(part_type_rad=4):
    """
    :param half_mass_radius: Use half mass radius instead of virial radius
    :param part_type_rad: The particle type to use for the half-mass-radius
    :return:
    """
    subfind.convert_to_cMpch()
    # Import gas
    coordinates, sml, m, XHI, group, subgroup = import_hdf5_folder(ProjectFiles.z3Snapshot, 0, hdf_paths=['Coordinates', 'SmoothingLength', 'Mass', 'SmoothedElementAbundance/Hydrogen', 'GroupNumber', 'SubGroupNumber'])
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]
    coordinates = None
    del coordinates
    nF = import_hdf5_folder(ProjectFiles.z3Urchin, hdf_paths=['HydrogenOneFraction'])[0]

    print len(x), "particles."
    # Find gas only in subhaloes
    # 1. Use haloes provided in the Subfind files to cut out gas particles in galaxies (==subhaloes) and haloes (FoF haloes)
    x_subhalo = []
    y_subhalo = []
    z_subhalo = []
    h_subhalo = []
    m_subhalo = []
    nF_subhalo = []
    XHI_subhalo = []
    grp_subhalo = []
    subgrp_subhalo = []

    # 1.1. Iterate over subhaloes, using group and subgroup numbers to narrow down particles
    for i in xrange(0, len(subfind.subhalo_centre_x), 1):

            # 1.1.1. Find particles in the subhalos
            #idxs_group = np.where(subfind.subhalo_group_number[i] == group)                         # All particles in the group (indexes)
            idxs_subgroup = np.where(subfind.subhalo_subgroup_number[i] == subgroup)#[idxs_group])    # All particles in subhalos in the group (indexes)
            #idxs_subgroup = idxs_group[0][idxs_subgroup]
            print "SubgroupNumber: ", subfind.subhalo_subgroup_number[i]
            #print idxs_subgroup
            # 1.1.2. Find only particles that are inside the half-mass-radius
            distance_squared = (x[idxs_subgroup] - subfind.subhalo_centre_x[i] )**2 \
                             + (y[idxs_subgroup] - subfind.subhalo_centre_y[i] )**2 \
                             + (z[idxs_subgroup] - subfind.subhalo_centre_z[i] )**2


            idxs_inside = np.where(distance_squared  <= subfind.subhalo_half_mass_rad[:, part_type_rad][i]**2) # + (sml[idxs_subgroup])**2
            idxs_inside = idxs_subgroup[0][idxs_inside]

            # 1.1.3 Add new particles to a list which will be plotted
            x_subhalo.extend(x[idxs_inside])
            y_subhalo.extend(y[idxs_inside])
            z_subhalo.extend(z[idxs_inside])
            h_subhalo.extend(sml[idxs_inside])

            m_subhalo.extend(m[idxs_inside])
            nF_subhalo.extend(nF[idxs_inside])
            XHI_subhalo.extend(XHI[idxs_inside])
            grp_subhalo.extend(group[idxs_inside])
            subgrp_subhalo.extend(subgroup[idxs_inside])

    print len(x_subhalo), "particles in subhalos found. Beginning plotting of results."

    # Save results
    folder = "HalfMassRad/"

    np.save("Galaxies/GasArrays/"+folder+"x"+str(part_type_rad), x_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"y"+str(part_type_rad), y_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"z"+str(part_type_rad), z_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"h"+str(part_type_rad), h_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"m"+str(part_type_rad), m_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"nF"+str(part_type_rad), nF_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"XHI"+str(part_type_rad), XHI_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"grp"+str(part_type_rad), grp_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"subgrp"+str(part_type_rad), subgrp_subhalo)


def index_parameters(p, indexes):
    if type(p) != list:
        p = [p]
    for i in xrange(0, len(p), 1):
        #print len(p[i])
        p[i] = p[i][indexes]
    return p	


# Save particles that are inside the half mass radius to arrays.
def isolate_gas_galaxies(save_folder = "HalfMassRad/", part_type_rad=4):
    """ Finds all the gas in galaxies (assumed to be spherical and have a radius = half-mass-radius), then save the remaining particles.
    :parameter: part_type_rad: The particle type to use for the half-mass-radius (0 for gas, 1 for dark matter, 4 for stars, 5 for black holes) """
    subfind.convert_to_cMpch()		# Make sure the radii are in the right units.
	
    # Import gas
    coordinates, sml, m, XHI, group, subgroup = import_hdf5_folder(ProjectFiles.z3Snapshot, 0, hdf_paths=['Coordinates', 'SmoothingLength', 'Mass', 'SmoothedElementAbundance/Hydrogen', 'GroupNumber', 'SubGroupNumber'])
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]
    coordinates = None
    del coordinates
    nF = import_hdf5_folder(ProjectFiles.z3Urchin, hdf_paths=['HydrogenOneFraction'])[0]
    print "Total particles: ", len(x)

	# Convert radii and distances (x,y, h) to same units
    # Have converted radii in subfind file when convert to cMpch is called

    # Must have [x,y,z,group, subgroup, ..., ...]
    parameters = [ x ,  y ,  z ,  group ,  subgroup ,  m , sml,  nF ,  XHI ]
    save_names = ["x", "y", "z", "group", "subgroup", "m", "h", "nF", "XHI"]
	
	# Eliminate null-group numbers (particles not in a group)
    parameters = index_parameters(parameters, np.where(parameters[3] != 1073741824))
    print "Grouped particles: ", len(parameters[0]), len(parameters[1]), len(parameters[2]), len(parameters[3]), len(parameters[len(parameters)-1])

	# Eliminate null-subgroup numbers (particles in a group, that are not in a subgroup)
    parameters = index_parameters(parameters, np.where(parameters[4] != 1073741824))
    print "SubGrouped particles: ", len(parameters[0]), len(parameters[1]), len(parameters[2]), len(parameters[3]), len(parameters[len(parameters)-1])
    # Create arrays to save final information in.
    saved_parameters = []
    for k in xrange(len(parameters)):
        saved_parameters.append([])

	# Create unique group numbers
    unique_groups = list(set(subfind.subhalo_group_number))
    unique_groups.sort()
    # print unique_groups
	
    # 1.1. Iterate over groups
    for i in xrange(0, len(unique_groups), 1):
        #print "Group", unique_groups[i]
        particle_group_indexes = np.where(unique_groups[i] == parameters[3])					# Indexes of particles in group i.
        particle_group_num 	= np.copy(parameters[3])[particle_group_indexes]					# Group numbers of particles in group i
        particle_subgroup_num  = np.copy(parameters[4])[particle_group_indexes]					# Subgroup numbers of particles in group i
        subhalo_indexes 	= np.where(subfind.subhalo_group_number == unique_groups[i])		# Indexes of sub-halos with group number i
        subhalos_in_group	= np.copy(subfind.subhalo_subgroup_number)[subhalo_indexes]			# Subgroup numbers of sub-halos in group i

        number_of_particles_in_group = len(particle_group_indexes[0])
        running_count = 0
        # 1.2. Iterate over it's subgroups
        for j in xrange(0, len(subhalos_in_group), 1):

            # 1.3. Find all particles with matching group AND subgroup numbers
            particle_subgroup_indexes = np.where(subhalos_in_group[j] == particle_subgroup_num) # Indexes of particles in subhalo j (relative to group i list)
            #print "Subgroup", subhalos_in_group[j], "# particles", len(particle_subgroup_indexes[0])
            running_count += len(particle_subgroup_indexes[0])
            subgroup_indexes = particle_group_indexes[0][particle_subgroup_indexes]				# Indexes of particles in group i and in subhalo j
            group_parameters = index_parameters(list(parameters), subgroup_indexes)
            distance_squared 	= (group_parameters[0] - subfind.subhalo_centre_x[subhalo_indexes][j] )**2 \
                                + (group_parameters[1] - subfind.subhalo_centre_y[subhalo_indexes][j] )**2 \
                                + (group_parameters[2] - subfind.subhalo_centre_z[subhalo_indexes][j] )**2
            # !! PROBLEM
            #print np.mean(distance_squared), subfind.subhalo_half_mass_rad[:, part_type_rad][subhalo_indexes][j]**2
            idxs_inside = np.where(distance_squared  <= subfind.subhalo_half_mass_rad[:, part_type_rad][subhalo_indexes][j]**2) # + (sml[idxs_subgroup])**2
            inside_parameters = index_parameters(list(group_parameters), idxs_inside)

            # 1.5. Store parameters for particles inside half mass radius
            for l in xrange(len(inside_parameters)):
                saved_parameters[l].extend(inside_parameters[l])
                #print inside_parameters[l]
    # 1.6. Save a copy of the final result to HDD
    for i in xrange(0, len(saved_parameters), 1):
        print "saving to ", "Galaxies/GasArrays/"+save_folder+save_names[i]+str(part_type_rad)
        np.save("Galaxies/GasArrays/"+save_folder+save_names[i]+str(part_type_rad), saved_parameters[i])
    print "Files saved. "

# Brute force: Loop over particles method (very slow!)
def isolate_gas_galaxies_brute(part_type_rad=4):
    """
    :param half_mass_radius: Use half mass radius instead of virial radius
    :param part_type_rad: The particle type to use for the half-mass-radius
    :return:
    """
    subfind.convert_to_cMpch()
    # Import gas
    coordinates, sml, m, XHI, group, subgroup = import_hdf5_folder(ProjectFiles.z3Snapshot, 0, hdf_paths=['Coordinates', 'SmoothingLength', 'Mass', 'SmoothedElementAbundance/Hydrogen', 'GroupNumber', 'SubGroupNumber'])
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]
    coordinates = None
    del coordinates
    nF = import_hdf5_folder(ProjectFiles.z3Urchin, hdf_paths=['HydrogenOneFraction'])[0]

    print len(x), "particles."
    # Find gas only in subhaloes
    # 1. Use haloes provided in the Subfind files to cut out gas particles in galaxies (==subhaloes) and haloes (FoF haloes)
    x_subhalo = []
    y_subhalo = []
    z_subhalo = []
    h_subhalo = []
    m_subhalo = []
    nF_subhalo = []
    XHI_subhalo = []
    grp_subhalo = []
    subgrp_subhalo = []

    # 1.1. Loop over particles, then loop over subhalos (galaxies)
    for i in xrange(0, len(x), 1):
        for j in xrange(0, len(subfind.subhalo_centre_x), 1):
                # 1.1.2. Find only particles that are inside the half-mass-radius
                distance_squared = (x[i] - subfind.subhalo_centre_x[j])**2 \
                                 + (y[i] - subfind.subhalo_centre_y[j])**2 \
                                 + (z[i] - subfind.subhalo_centre_z[j])**2

                if distance_squared <= subfind.subhalo_half_mass_rad[:, part_type_rad][j]**2:
                    # 1.1.3 Add new particles to a list which will be plotted
                    x_subhalo.append(x[i])
                    y_subhalo.append(y[i])
                    z_subhalo.append(z[i])
                    h_subhalo.append(sml[i])
                    m_subhalo.append(m[i])
                    nF_subhalo.append(nF[i])
                    XHI_subhalo.append(XHI[i])
                    grp_subhalo.append(group[i])
                    subgrp_subhalo.append(subgroup[i])

    print len(x_subhalo), "particles in subhalos found. Beginning plotting of results."

    # Save results
    folder = "HalfMassRad/"

    np.save("Galaxies/GasArrays/"+folder+"x"+str(part_type_rad), x_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"y"+str(part_type_rad), y_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"z"+str(part_type_rad), z_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"h"+str(part_type_rad), h_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"m"+str(part_type_rad), m_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"nF"+str(part_type_rad), nF_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"XHI"+str(part_type_rad), XHI_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"grp"+str(part_type_rad), grp_subhalo)
    np.save("Galaxies/GasArrays/"+folder+"subgrp"+str(part_type_rad), subgrp_subhalo)


# Save particles that are inside the virial radius to arrays.
def isolate_gas_galaxies_and_halos():
    subfind.convert_to_cMpch()
    folder = "VirialRad/"
    # Import gas
    coordinates, sml, m, XHI, group, subgroup = import_hdf5_folder(ProjectFiles.z3Snapshot, 0, hdf_paths=['Coordinates', 'SmoothingLength', 'Mass', 'SmoothedElementAbundance/Hydrogen', 'GroupNumber', 'SubGroupNumber'])
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]
    coordinates = None
    del coordinates
    nF = import_hdf5_folder(ProjectFiles.z3Urchin, hdf_paths=['HydrogenOneFraction'])[0]

    print len(x), "particles."
    # Find gas only in subhaloes
    # 1. Use haloes provided in the Subfind files to cut out gas particles in galaxies (==subhaloes) and haloes (FoF haloes)
    x_halo = []
    y_halo = []
    z_halo = []
    h_halo = []
    m_halo = []
    nF_halo = []
    XHI_halo = []
    grp_halo = []
    subgrp_halo = []

    # 1.1. Iterate over subhaloes, using group and subgroup numbers to narrow down particles
    for i in xrange(0, len(subfind.halo_centre_x), 1):

            # 1.1.1. Find particles in the halos
            idxs_group = np.where(subfind.subhalo_group_number[i] == group)                         # All particles in the group (indexes)
            #idxs_subgroup = np.where(subfind.subhalo_subgroup_number[i] == subgroup[idxs_group])    # All particles in subhalos in the group (indexes)
            #idxs_subgroup = idxs_group[0][idxs_subgroup]

            # 1.1.2. -- OR --  Find only particles that are inside the virial-radius
            distance_squared = (x[idxs_group] - subfind.halo_centre_x[i] )**2 \
                 + (y[idxs_group] - subfind.halo_centre_y[i] )**2 \
                 + (z[idxs_group] - subfind.halo_centre_z[i] )**2


            idxs_inside = np.where(distance_squared  <= subfind.halo_virial_rad[i]**2) # + (sml[idxs_subgroup])**2
            #idxs_inside = idxs_subgroup[idxs_inside]

            # 1.1.3 Add new particles to a list which will be plotted
            x_halo.extend(x[idxs_inside])
            y_halo.extend(y[idxs_inside])
            z_halo.extend(z[idxs_inside])
            h_halo.extend(sml[idxs_inside])

            m_halo.extend(m[idxs_inside])
            nF_halo.extend(nF[idxs_inside])
            XHI_halo.extend(XHI[idxs_inside])
            grp_halo.extend(group[idxs_inside])
            subgrp_halo.extend(subgroup[idxs_inside])

    print len(x_halo), "particles in subhalos found. Beginning plotting of results."

    # Save results
    np.save("Galaxies/GasArrays/"+folder+"x", x_halo)
    np.save("Galaxies/GasArrays/"+folder+"y", y_halo)
    np.save("Galaxies/GasArrays/"+folder+"z", z_halo)
    np.save("Galaxies/GasArrays/"+folder+"h", h_halo)
    np.save("Galaxies/GasArrays/"+folder+"m", m_halo)
    np.save("Galaxies/GasArrays/"+folder+"nF", nF_halo)
    np.save("Galaxies/GasArrays/"+folder+"XHI", XHI_halo)
    np.save("Galaxies/GasArrays/"+folder+"grp", grp_halo)
    np.save("Galaxies/GasArrays/"+folder+"subgrp", subgrp_halo)


# Plot the x and y positions of all the saved particles
def plot_isolated_gas(folder="VirialRad", part_type=None):
    if part_type is None:
        x = np.load("Galaxies/GasArrays/"+folder+"/x.npy")
        y = np.load("Galaxies/GasArrays/"+folder+"/y.npy")
    else:
        x = np.load("Galaxies/GasArrays/"+folder+"/x"+str(part_type)+".npy")
        y = np.load("Galaxies/GasArrays/"+folder+"/y"+str(part_type)+".npy")

    # 1.2 Plot results
    fig = plt.figure()

    # 1.2.1. Plot the particles only in subhaloes
    plt.scatter(x, y, s=2, c='blue', marker='o')

    # 1.2.2. Plot half-mass-radius of sub-haloes
    for j in xrange(0, len(subfind.subhalo_centre_x), 1):
        circle = plt.Circle((subfind.subhalo_centre_x[j], subfind.subhalo_centre_y[j]), radius=subfind.subhalo_half_mass_rad[:,0][j],
                            facecolor="none", edgecolor='black', linestyle='solid')
        fig.gca().add_artist(circle)


    # 1.2.3. Plot virial radius of FOF haloes
    for i in xrange(0, len(subfind.halo_centre_x)):
        circle = plt.Circle((subfind.halo_centre_x[i], subfind.halo_centre_y[i]), radius=subfind.halo_virial_rad[i],
                            facecolor="none", edgecolor='black', alpha=0.7, linestyle='dotted', linewidth=1)
        fig.gca().add_artist(circle)


    plt.xlim((0, Snapshot.BoxsizecMpch))
    plt.ylim((0, Snapshot.BoxsizecMpch))

    plt.show()


# Create the galaxy grid ===================================================================================================================================== #


def create_grid_gal(save_name="Galaxies/z3GalaxyGridHalfMassRad", part_type=4):
    folder = "Galaxies/GasArrays/HalfMassRad/"
    x = np.load(folder+"x"+str(part_type)+".npy")
    y = np.load(folder+"y"+str(part_type)+".npy")
    h = np.load(folder+"h"+str(part_type)+".npy")
    m = np.load(folder+"m"+str(part_type)+".npy")
    nF = np.load(folder+"nF"+str(part_type)+".npy")
    XHI = np.load(folder+"XHI"+str(part_type)+".npy")
    print len(x), len(y), len(h), len(m), len(nF), len(XHI)

    #print x
    grid, cell_length = create_grid(x, y, h, [nF, XHI, m])
    grid, cell_length = convert_cell_to_cgs(grid, cell_length)
    save_grid(grid, save_name+str(part_type))
    print np.min(grid), np.max(grid)
    subfind.convert_to_pkpc()
    plot_grid(grid, save_name+str(part_type), halos=True,halo_x=subfind.subhalo_centre_x, halo_y=subfind.subhalo_centre_y, halo_radii=subfind.subhalo_half_mass_rad[:, part_type])
	

def plot_galaxy_grid(file_name="Galaxies/GasArrays/z3GalaxiesNGrids3RMin128RMax4096Grid.npy", save_name = "Galaxies/z3GalaxyGrid3", part_type=4, half_mass_rad=False, virial_rad=False):
    subfind.convert_to_pkpc()
    Snapshot.importSnap(ProjectFiles.z3Snapshot, ProjectFiles.z3Urchin)
    #print "Snapshot info: ", Snapshot.Redshift, Snapshot.BoxsizecMpch
    

    grid = np.load(file_name)

    cell_to_pkpc = Snapshot.Boxsizepkpc/grid.shape[0]

    extent  = (0, grid.shape[0]*cell_to_pkpc, 0, grid.shape[1]*cell_to_pkpc)
    x_units = r"$ \rm pkpc$"

    if np.min(grid) <= 0:
        grid = ma.masked_where(grid <= 0, grid)
    colour_map = colour_map_split3(np.log10(grid))#, colour_map_splits[0],colour_map_splits[1])

    fig = plt.figure()
    plt.imshow(np.log10(grid), origin='lower', interpolation='none', cmap=colour_map, extent=extent)
    cbar = plt.colorbar()
    cbar.set_label(r"$ \rm log_{10} \ N_{HI} $", rotation=-90, labelpad=30)
    plt.xlabel(r"$ \rm x$"+"$ \ ($"+x_units+"$)$")
    plt.ylabel(r"$ \rm y$"+"$ \ ($"+x_units+"$)$")

    # Half-mass radius
    if half_mass_rad:
        for j in xrange(0, len(subfind.subhalo_centre_x), 1):
            circle = plt.Circle((subfind.subhalo_centre_x[j], subfind.subhalo_centre_y[j]), radius=subfind.subhalo_half_mass_rad[:,part_type][j],
                                facecolor="none", edgecolor='black', linestyle='solid')
            fig.gca().add_artist(circle)

    # Virial radius
    if virial_rad:
        for i in xrange(0, len(subfind.halo_centre_x)):
            circle = plt.Circle((subfind.halo_centre_x[i], subfind.halo_centre_y[i]), radius=subfind.halo_virial_rad[i],
                                facecolor="none", edgecolor='black', alpha=0.7, linestyle='dotted', linewidth=1)
            fig.gca().add_artist(circle)

    plt.savefig(save_name+str(part_type)+".pdf", dpi=1000, bbox_inches='tight')
    # Close up of rightmost over density
    plt.xlim((2700,3100))
    plt.ylim((700, 1100))
    plt.savefig(save_name+str(part_type)+"CloseUp.pdf", dpi=1000, bbox_inches='tight')
    plt.show()


def plot_rings_on_grid(part_type_rad=4):
    save_name = "z3FullGridAndRings"+str(part_type_rad)
    grid = np.load(ProjectFiles.z3gridHI)

    # Choose units for imshow and radii
    cell_to_pkpc = Snapshot.Boxsizepkpc/grid.shape[0]
    subfind.convert_to_pkpc()

    extent  = (0, grid.shape[0]*cell_to_pkpc, 0, grid.shape[1]*cell_to_pkpc)
    x_units = r"$ \rm pkpc$"

    if np.min(grid) <= 0:
        grid = ma.masked_where(grid <= 0, grid)
    colour_map = colour_map_split3(np.log10(grid))#, colour_map_splits[0],colour_map_splits[1])

    fig = plt.figure()
    plt.imshow(np.log10(grid), origin='lower', interpolation='none', cmap=colour_map, extent=extent)
    cbar = plt.colorbar(pad=0)
    cbar.set_label(r"$ \rm log_{10} \ (N_{HI}/cm^{-2}) $", rotation=-90, labelpad=15)
    plt.xlabel(r"$ \rm x$"+"$ \ ($"+x_units+"$)$", labelpad=3)
    plt.ylabel(r"$ \rm y$"+"$ \ ($"+x_units+"$)$", labelpad=2)


    # Half-mass radius
    for j in xrange(0, len(subfind.subhalo_centre_x), 1):
        circle = plt.Circle((subfind.subhalo_centre_x[j], subfind.subhalo_centre_y[j]), radius=subfind.subhalo_half_mass_rad[:,part_type_rad][j],
                            facecolor="none", edgecolor='black', linestyle='solid')
        fig.gca().add_artist(circle)

    # Virial radius:
    for i in xrange(0, len(subfind.halo_centre_x)):
        circle = plt.Circle((subfind.halo_centre_x[i], subfind.halo_centre_y[i]), radius=subfind.halo_virial_rad[i],
                            facecolor="none", edgecolor='black', alpha=0.7, linestyle='dotted', linewidth=1)
        fig.gca().add_artist(circle)

    plt.savefig(save_name+save_name_add1+".pdf", dpi=1000, bbox_inches='tight', pad_inches=0.04)
    plt.savefig(save_name+save_name_add1+".png", dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.savefig(save_name+save_name_add1+".jpg", dpi=300, bbox_inches='tight', pad_inches=0.04)

    # Close up of rightmost over density
    plt.xlim((2700,3100))
    plt.ylim((700, 1100))
    plt.xticks([2700, 2800, 2900, 3000, 3100])
    plt.yticks([700, 800, 900, 1000, 1100])

    plt.savefig(save_name+save_name_add1+"CloseUp.pdf", dpi=1000, bbox_inches='tight', pad_inches=0.04)
    plt.savefig(save_name+save_name_add1+"CloseUp.png", dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.savefig(save_name+save_name_add1+"CloseUp.jpg", dpi=300, bbox_inches='tight', pad_inches=0.04)

    plt.show()

	
# Plot overlay =============================================================================================================================================== #


def plot_overlay(grid1, grid2, x_limit=(2600, 3100), y_limit=(700,1200),save_name="Galaxies/OverlayHalfMass"):
    separate_plots = 0
    just_lls = 1

    extent = (0, Snapshot.Boxsizepkpc, 0, Snapshot.Boxsizepkpc)
    x_units = r"$ \rm pkpc$"
    colour_map = colour_map_split3(np.log10(grid1))

    print "Grid1", np.min(grid1), np.max(grid1)
    print "Grid2", np.min(grid2), np.max(grid2)

    # Mask zeros
    grid2 = ma.masked_where(grid2 == 0, grid2)
    print "Minimum of masked grid2", np.min(grid2)
    print "Minimum of masked grid2", np.min(np.log10(grid2))

    if just_lls:
        grid1 = mask_all_but_LLS(grid1, False)
        colour_map = colour_map_split1(white_blue=False, green_yellow=True)

    if separate_plots:
        fig = plt.figure()
        ax = plt.subplot(1, 2, 1)
        plt.imshow(np.log10(grid1), origin='lower', extent=extent, cmap=colour_map)
        cbar = plt.colorbar()
        ax.set_xlim((x_limit))
        ax.set_ylim((y_limit))


        ax2 = plt.subplot(1, 2, 2)
        colour_map = colour_map_split3(np.log10(grid2))

        plt.imshow(np.log10(grid2), origin='lower', extent=extent, cmap=colour_map)
        cbar = plt.colorbar()
        ax2.set_xlim((x_limit))
        ax2.set_ylim((y_limit))
        plt.show()

    # Plot on same plot
    fig = plt.figure()

    # galaxies
    colour_map = colour_map_split3(np.log10(grid2))
    colour_map.set_bad('white')

    #colour_map.set_clim(12, 23)
    plt.imshow(np.log10(grid2), origin='lower', extent=extent, cmap=colour_map, alpha = 1.0)
    cbar = plt.colorbar(pad=0)
    cbar.set_label(r"$ \rm log_{10} \ N_{HI} $", rotation=-90, labelpad=30)
    plt.xlabel(r"$ \rm x$"+"$ \ ($"+x_units+"$)$")
    plt.ylabel(r"$ \rm y$"+"$ \ ($"+x_units+"$)$")                          # LLSs

    #colour_map = colour_map_split3(np.log10(grid1))
    custom_map = custom_map1()
    plt.imshow(np.log10(grid1), origin='lower', extent=extent, cmap=custom_map, alpha = 0.6)


    subfind.convert_to_pkpc()

    # Add virial radius rings
    #for i in xrange(0, len(subfind.halo_centre_x)):
    #    circle = plt.Circle((subfind.halo_centre_x[i], subfind.halo_centre_y[i]), radius=subfind.halo_virial_rad[i],
    #                        facecolor="none", edgecolor='black', alpha=0.6, linestyle='solid', linewidth=1)
    #    fig.gca().add_artist(circle)

    # Half-mass radius
    print subfind.units
    part_type_rad = 4
    for j in xrange(0, len(subfind.subhalo_centre_x), 1):
        circle = plt.Circle((subfind.subhalo_centre_x[j], subfind.subhalo_centre_y[j]), radius=subfind.subhalo_half_mass_rad[:, part_type_rad][j],
                            facecolor="none", edgecolor='black', linestyle='solid')
        fig.gca().add_artist(circle)


    plt.savefig(save_name+".pdf", dpi=1000, bbox_inches='tight')
    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.savefig(save_name+"CloseUp.pdf", dpi=1000, bbox_inches='tight')
    plt.show()

	
# Test code ================================================================================================================================================== #


if __name__ == '__main__':

    # Plot overlay of haloes and LLS
    #grid_LLS = np.load(ProjectFiles.z3gridHI)
    #grid_halos = np.load("Galaxies/Halos.npy")
    #plot_overlay(grid_LLS,grid_halos,save_name="Galaxies/OverlayLLSandHalos")

    #grid = np.load(ProjectFiles.z3gridHI)
    #grid_gal = np.load(ProjectFiles.z3gridAllSubhalos)

    #plot_overlay(grid, grid_gal)

    # Basic: Keep particles with a sub-group number and project the rest.



    # 1. Find gas inside virial radius (galaxy + haloes)
    #isolate_gas_galaxies_and_halos()
    #plot_isolated_gas()

    # 2 Find gas inside half-mass-radius of stars
    #isolate_gas_galaxies(part_type_rad=4)
    #create_grid_gal(part_type=4)
    #plot_galaxy_grid("Galaxies/z3GalaxyGridHalfMassRad.npy", save_name="Galaxies/z3GalaxiesHalfMassRadGrid.npy", part_type=4, half_mass_rad=True)

    # 2.5 Find gas inside half-mass radius of gas
    #isolate_gas_galaxies(part_type_rad=0)
    #create_grid_gal(part_type=0)
    #plot_galaxy_grid("Galaxies/z3GalaxyGridHalfMassRad.npy", save_name="Galaxies/z3GalaxiesHalfMassRadGrid.npy", part_type=0, half_mass_rad=True)

    #plot_isolated_gas("HalfMassRad", 4)

    # 3. plot results
    plot_rings_on_grid()
    #plot_galaxy_grid()