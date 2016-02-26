""" Contains all miscellaneous functions that I created and used in my level 4 project. """
from __future__ import division
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import colors

__author__ = 'Matt Law'


# Number/Array processing ==================================================================================================================================== #
def rd_to_int(num):
    """ Perform standard rounding on 'num' to integer type. If >= .5 then round up, if < .5 round down. """
    diff = num - int(num)
    if diff >= 0.5:
        return int(num)+1
    else:
        return int(num)


def downsize(data, fraction):                                               # Randomly down size data
    """ Pseudo-randomly down-size data.
    :param data: A list of data-lists (with equal lengths), e.g. [[1,2,3,4],[5,6,7,8]]
    :param fraction: The amount to down size the data-lists by; between 0 and 1
    :return: Randomly down-sized list of lists, e.g. [[1,2].[5,6]] for fraction = 0.5
    """
    idxs = np.array([np.random.randint(i/fraction,(i+1)/fraction,1)[0] for i in range(int(len(data[0])*fraction))])
    if fraction*len(data[0]) <= 1.0:
        print "Too little data to downsize. Returning full array of data."
        return data
    else:
        for i in range(len(data)):
            data[i] = data[i][idxs]
        return data


def convert_list_of_lists_to_list(lol):		
	""" Take [[a,b,..],[c,d,e,....],[f,...],[g,...]] and convert it to [a,b,..,c,d,e,....,f,...,g,...]. """
	new_list = []
	for i in xrange(0, len(lol), 1):
		new_list.extend(lol[i])
	return new_list
		
		
# Sorting ============================================================================================================ #
def sort2(s1, s2):
    """ Sort sequence one "s1" into ascending order, whilst
    maintaining it's original mapping with sequences;"s2", "s3", ..etc. """
    idxs = np.argsort(s1)
    s2 = s2[idxs]
    s1 = s1[idxs]
    return s1, s2


def sortN(sequences):
    """ Sort sequence one "s1" into ascending order, whilst
    maintaining it's original mapping with sequences; "s2", "s3", ..etc.
    s2 should be a list of arrays/lists, e.g. [[1,2,3],[4,5,6],[3,2,5], ....]"""
    idxs = np.argsort(sequences[0])
    for i in xrange(0, len(sequences), 1):
        sequences[i] = sequences[i][idxs]
    return sequences


def unique_elements(seq):
    """ Find and return the unique elements in a list."""
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if not (x in seen or seen_add(x))]


def find_nearest(array, value):
    """ Returns the index of the element of "array" which is closest to "value". """
    array = np.array(array)
    diff = np.abs(array-value)

    if array.ndim == 1:
        idx = diff.argmin()
        return idx

    if array.ndim == 2:
        minima = []
        ith    = []
        for i in range(len(diff[0,:])):         # Iterate over the 1st ROW!!
            i_idx = np.argmin(diff[:, i])       # Find minima of consecutive columns
            minima.append(diff[:, i][i_idx])    # Record minima value
            ith.append(i_idx)                   # Record index of minima
        j_idx = np.argmin(minima)               # i_idx is the ith coordinate of the true minima
        i_idx = ith[j_idx]
        return i_idx, j_idx

    if array.ndim == 3:
        j_idxs=[]
        k_idxs=[]
        minima=[]
        for i in range(len(diff[:,0,0])):       # jk_plane = diff[i, :, :]
            p_idx = []                          # Find minimia of j-k plane
            p_minima = []
            for j in range(len(diff[i,:,0])):   # Iterate over j (down columns)!!
                k_idx = np.argmin(diff[i,j,:])
                p_minima.append(diff[i,j,:][k_idx])
                p_idx.append(k_idx)
            j_idx = np.argmin(p_minima)
            k_idx = p_idx[j_idx]
            j_idxs.append(j_idx)
            k_idxs.append(k_idx)
            minima.append(diff[i,:,:][j_idx, k_idx])
        i_idx = np.argmin(minima)
        j_idx = j_idxs[i_idx]
        k_idx = k_idxs[i_idx]
        return i_idx, j_idx, k_idx


# Plotting =========================================================================================================== #
def colour_map_split3(data, split1=17.2, split2=20.3):
    minimum = np.min(data)
    maximum = np.max(data)

    # If maximum < split1 only return bottom part
    if maximum <= split1:
        return colour_map_split1()
    elif minimum >= split2:
        colour_map_split1(red=True)
    elif minimum >= split1 and maximum <= split2:
        return colour_map_split1(green_yellow=True)

    elif minimum >= split1:
        return colour_map_split2(data, split = split2)

    LLS_colour = (split1 - minimum) / (maximum - minimum)
    DLA_colour = (split2 - minimum) / (maximum - minimum)
    #print LLS_colour, DLA_colour
    print "Data range     : ", minimum, maximum
    print "Colour map from: ", 0.0, LLS_colour, DLA_colour, 1.0
    cdict = {'red': ((0.0, 1.0, 1.0),            # White
                     (LLS_colour, 0.0, 0.0),     # Blue
                     (DLA_colour, 0.95, 0.95),   # ( position in colourbar (from 0 to 1) , value of 'color' to interpolate from below up to the point , value of color to interpolate up from )
                     (1.0, 0.5, 1.0)),
            'green': ((0.0, 1.0, 1.0),
                     (LLS_colour, 0.0, 1.0),
                     (DLA_colour, 0.95, 0.0),
                      (1.0, 0.0, 0.0)),
            'blue': ((0.0, 1.0, 1.0),
                     (LLS_colour, 0.95, 0.0),
                     (DLA_colour, 0.3, 0.0),
                        (1.0, 0.0, 0.0))}

    my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return my_cmap


def colour_map_split2(data, split=17.2, blue_green=False):
    minimum = np.min(data)
    maximum = np.max(data)
    LLS_colour = (split - minimum) / (maximum - minimum)
    print "Data range     : ", minimum, maximum
    print "Colour map from: ", 0.0, LLS_colour, 1.0
    if blue_green:
        cdict = {'red': ((0.0, 1.0, 1.0),            # White
                         (LLS_colour, 0.0, 0.0),     # Blue
                         (1.0, 0.5, 1.0)),
                'green': ((0.0, 0.0, 1.0),
                         (LLS_colour, 0.0, 1.0),
                         (1.0,  0.95, 0.0)),
                'blue': ((0.0, 1.0, 1.0),
                         (LLS_colour, 0.95, 0.0),
                         (1.0, 0.3, 0.0))}

    else:   # Green to red
        cdict = {'red': ((0.0, 0.0, 0.0),            # White
                         (LLS_colour, 0.95, 0.95),   # Blue
                         (1.0, 0.95, 0.95)),
                'green': ((0.0, 1.0, 1.0),
                         (LLS_colour, 0.95, 0.0),
                         (1.0,  0.0, 0.0)),
                'blue': ((0.0, 0.95, 0.0),
                         (LLS_colour,  0.3, 0.0),
                         (1.0, 0.0, 0.0))}

    my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return my_cmap


def colour_map_split1(white_blue=False, green_yellow=False, red=False):

    # White to Blue
    if white_blue:
        cdict = {'red': ((0.0, 1.0, 1.0),
                         (1.0, 0.0, 0.0)),
                'green': ((0.0, 1.0, 1.0),
                         (1.0,  0.0, 1.0)),
                'blue': ((0.0, 1.0, 1.0),
                         (1.0, 0.95, 0.0))}

    # Green to yellow
    elif green_yellow:
        cdict = {'red': ((0.0, 0.0, 0.0),
                         (1.0, 0.95, 0.95)),
                'green': ((0.0, 0.0, 1.0),
                         (1.0,  0.95, 0.0)),
                'blue': ((0.0, 0.95, 0.1),
                         (1.0, 0.3, 0.0))}
    else:
        cdict = {'red':((0.0, 0.95, 0.95),   # ( position in colour bar (from 0 to 1) , value of 'color' to interpolate from below up to the point , value of color to interpolate up from )
                         (1.0, 0.5, 1.0)),
                'green':((0.0, 0.95, 0.0),
                         (1.0, 0.0, 0.0)),
                'blue': ((0.0, 0.3, 0.0),
                         (1.0, 0.0, 0.0))}

    my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return my_cmap


def custom_map1():
    # Black
    cdict = {'red': ((0.0, 0.0, 0.0),
                     (1.0, 0.0, 0.0)),
            'green': ((0.0, 0.0, 0.0),
                     (1.0,  0.0, 0.0)),
            'blue': ((0.0, 0.0, 0.0),
                     (1.0, 0.0, 0.0))}
    # White
    '''
    cdict = {'red': ((0.0, 1.0, 1.0),
                     (1.0, 1.0, 1.0)),
            'green': ((0.0, 1.0, 1.0),
                     (1.0,  1.0, 1.0)),
            'blue': ((0.0, 1.0, 1.0),
                     (1.0, 1.0, 1.0))}
    '''
    my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return my_cmap


def split_spectral():
    spectral_bottom = \
    {u'blue' : [(0.0, 0.6352941393852234, 0.6352941393852234),
                (0.1, 0.7411764860153198, 0.7411764860153198),
                (0.2, 0.6470588445663452, 0.6470588445663452),
                (0.3, 0.6431372761726379, 0.6431372761726379),
                (0.4, 0.5960784554481506, 0.5960784554481506),
                (0.5, 0.7490196228027344, 0.7490196228027344),
                (0.6, 0.545098066329956, 0.545098066329956),
                (0.7, 0.3803921639919281, 0.3803921639919281),
                (0.8, 0.26274511218070984, 0.26274511218070984),
                (0.9, 0.30980393290519714, 0.30980393290519714),
                (1.0, 0.25882354378700256, 0.25882354378700256)],
     u'green': [(0.0, 1.0, 1.0),
                (0.2, 0.8784313797950745, 0.8784313797950745),
                (0.4, 0.6823529601097107, 0.6823529601097107),
                (0.6, 0.4274509847164154, 0.4274509847164154),
                (0.8, 0.24313725531101227, 0.24313725531101227 ),
                (1.0, 0.003921568859368563, 0.003921568859368563 )],
     u'red'  : [(0.0, 1.0, 1.0),
                (0.2, 0.9960784316062927, 0.9960784316062927),
                (0.4, 0.9921568632125854, 0.9921568632125854),
                (0.6, 0.95686274766922  , 0.95686274766922),
                (0.8, 0.8352941274642944, 0.8352941274642944),
                (1.0, 0.6196078658103943, 0.6196078658103943)]}



    spectral_top = \
    {u'blue' : [(0.0, 0.25882354378700256, 0.25882354378700256),
                (0.1, 0.30980393290519714, 0.30980393290519714),
                (0.2, 0.26274511218070984, 0.26274511218070984),
                (0.3, 0.3803921639919281, 0.3803921639919281),
                (0.4, 0.545098066329956, 0.545098066329956),
                (0.5, 0.7490196228027344, 0.7490196228027344),
                (0.6, 0.5960784554481506, 0.5960784554481506),
                (0.7, 0.6431372761726379, 0.6431372761726379),
                (0.8, 0.6470588445663452, 0.6470588445663452),
                (0.9, 0.7411764860153198, 0.7411764860153198),
                (1.0, 0.6352941393852234, 0.6352941393852234)],
     u'green': [(0.0, 1.0, 1.0),
                (0.2, 0.9607843160629272, 0.9607843160629272),
                (0.4, 0.8666666746139526, 0.8666666746139526),
                (0.6, 0.7607843279838562, 0.7607843279838562),
                (0.8, 0.5333333611488342, 0.5333333611488342),
                (1.0, 0.30980393290519714, 0.30980393290519714)],
     u'red'  : [(0.0, 1.0, 1.0),
                (0.2, 0.9019607901573181, 0.9019607901573181),
                (0.4, 0.6705882549285889, 0.6705882549285889),
                (0.6, 0.4000000059604645, 0.4000000059604645),
                (0.8, 0.19607843458652496, 0.19607843458652496),
                (1.0, 0.3686274588108063, 0.3686274588108063)]}

    cmap1 = colors.LinearSegmentedColormap("Spectral_bottom", spectral_bottom)
    cmap2 = colors.LinearSegmentedColormap("Spectral_top", spectral_top)
    return cmap1, cmap2


def histogram(array, n_bins=200, remove_empty_bins=False):
    """ Create a histogram of the data, remove zeros, then return it.
    :param array: Any list or array
    :param n_bins: The number of equal sized bins to split array into
    :return: bin centers, frequency for each bin. """
    # Take histogram of data
    array = ma.masked_where(array == 0, array)
    hist, bins  = np.histogram(array, bins=n_bins)
    center      = (bins[:-1] + bins[1:]) / 2

    if remove_empty_bins:
        hist2 = []
        cent2 = []
        for i, j in enumerate(hist):                # Remove boring bins with nothing in
            if j != 0:
                hist2.append(j)
                cent2.append(center[i])
        center = np.array(cent2)
        hists = np.array(hist2)
    else:
        hists = hist

    return center, hists


def plot_overlay(grid1, grid2, x_limit=(2600, 3100), y_limit=(700,1200),save_name="Galaxies/OverlayHalfMass", extent=(0, 1, 0, 1)):
    """
    Produce a colour map of two grids, over-layed on top of each other (grid 2 on top of grid 1)
    :param grid1: 2D array
    :param grid2: 2D array
    :param x_limit: range on the x-axis to plot
    :param y_limit: range on the y-axis to plot
    :param save_name: file name to save plots as
    """
    separate_plots = 0
    just_lls = 0

    #extent = (0, Snapshot.Boxsizepkpc, 0, Snapshot.Boxsizepkpc)
    x_units = r"$ \rm pkpc$"
    colour_map = colour_map_split3(np.log10(grid1))

    print "Grid1", np.min(grid1), np.max(grid1)
    print "Grid2", np.min(grid2), np.max(grid2)

    # Mask zeros
    grid2 = ma.masked_where(grid2 == 0, grid2)
    print "Minimum of masked grid2", np.min(grid2)
    print "Minimum of masked grid2-log10", np.max(np.log10(grid2))

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


    # Under plot
    colour_map = colour_map_split3(np.log10(grid1))
    plt.imshow(np.log10(grid1), origin='lower', extent=extent, cmap=colour_map, alpha = 1.0)
    cbar = plt.colorbar(pad=0)
    cbar.set_label(r"$ \rm log_{10} \ (N_{HI}/cm^{-2}) $", rotation=-90, labelpad=15)
    plt.xlabel(r"$ \rm x$"+"$ \ ($"+x_units+"$)$", labelpad=3)
    plt.ylabel(r"$ \rm y$"+"$ \ ($"+x_units+"$)$", labelpad=2)

    # Over plot
    #colour_map = colour_map_split3(np.log10(grid2))
    #colour_map.set_bad('white')
    colour_map = custom_map1()
    #colour_map.set_clim(12, 23)
    plt.imshow(np.log10(grid2), origin='lower', extent=extent, cmap=colour_map, alpha = 0.4)

    #subfind.convert_to_pkpc()

    # Add virial radius rings
    #for i in xrange(0, len(subfind.halo_centre_x)):
    #    circle = plt.Circle((subfind.halo_centre_x[i], subfind.halo_centre_y[i]), radius=subfind.halo_virial_rad[i],
    #                        facecolor="none", edgecolor='black', alpha=0.6, linestyle='solid', linewidth=1)
    #    fig.gca().add_artist(circle)

    # Half-mass radius
    #print subfind.units
    #part_type_rad = 4
    #for j in xrange(0, len(subfind.subhalo_centre_x), 1):
    #    circle = plt.Circle((subfind.subhalo_centre_x[j], subfind.subhalo_centre_y[j]), radius=subfind.subhalo_half_mass_rad[:, part_type_rad][j],
    #                        facecolor="none", edgecolor='black', linestyle='solid')
    #    fig.gca().add_artist(circle)


    plt.savefig(save_name+".pdf", dpi=1000, bbox_inches='tight', pad_inches=0.04)
    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.savefig(save_name+"CloseUp.pdf", dpi=1000, bbox_inches='tight', pad_inches=0.04)
    plt.show()



# Array-based ======================================================================================================== #


def mask_zeros(grid):
    """ Mask all zeros in grid, and return the masked array. """
    grid = ma.masked_where(grid == 0, grid)
    return grid


def mask_leq_zero(grid):
    """ Mask all numbers in grid that are less than or equal to 0, and return the masked array. """
    grid = ma.masked_where(grid <= 0, grid)
    return grid


def mask_all_but_DLA(grid, mask_with_zero=True):
    """ Mask all non-DLA, and return the masked array. """
    grid = ma.masked_where(np.log10(grid) < 20.3, grid)
    if mask_with_zero:
        grid = grid.filled(0)
    return grid


def mask_all_but_LLS(grid, mask_with_zero=True):
    """ Mask all non-LLSs, and return the masked array. """
    grid = ma.masked_where(np.log10(grid) < 17.2, grid)                      # Isolate LLs
    grid = ma.masked_where(np.log10(grid) > 20.3, grid)
    if mask_with_zero:
        grid = grid.filled(0)
    return grid


def bin_data_x_space(x, y, bin_width = 10, xerr=0):
    """ Bins two arrays based on the value of x. Splits x into several bins of size bin_width (same units as x), and
    returns the mean values of x and y in each bin. It also calculates and returns the standard error on y, due to binning.
    :return: x, y, y_err """
    min_x, max_x = min(x), max(x)
    number_means = int(np.ceil( (max_x - min_x)/bin_width ))     # Number of x-points (mean values stored in bins)
    x_mean, y_mean, x_error, y_error = np.zeros(number_means), np.zeros(number_means), np.zeros(number_means), np.zeros(number_means)

    print "Binning between: ", min_x, max_x, "with ", number_means, "bins"

    for i in xrange(0, number_means, 1):
        indexes     = np.where(min_x + i*bin_width <= x)
        indexes2    = np.where(x[indexes] <= min_x + (i+1)*bin_width)

        x_bin       = x[indexes][indexes2]
        N           = len(x_bin)
        x_mean[i]   = x_bin.mean()
        x_error[i]  = np.sqrt( ( np.sum( (x_bin - x_mean[i])**2 ) + (xerr**2)*N ) / (N*(N-1)) )

        ybin        = y[indexes][indexes2]
        y_mean[i]   = ybin.mean(axis=0)
        y_error[i]  = np.sqrt( np.sum( (ybin - y_mean[i]) **2 ) /  ( N - 1 ) ) / np.sqrt(N)

    return x_mean, y_mean, x_error, y_error


def bin_data_x_space_total(x, y, bin_width = 10, xerr=0):
    """ Same as bin_data_x_space but instead of finding the mean in bins, it finds the total y in the bin.
    :return: x, y, y_err """
    min_x, max_x = min(x), max(x)
    number_means = int(np.ceil( (max_x - min_x)/bin_width ))     # Number of x-points (mean values stored in bins)
    x_mean, y_total, x_error, y_error = np.zeros(number_means), np.zeros(number_means), np.zeros(number_means), np.zeros(number_means)

    for i in xrange(0, number_means, 1):
        indexes     = np.where(min_x + i*bin_width <= x)
        indexes2    = np.where(x[indexes] <= min_x + (i+1)*bin_width)

        x_bin       = x[indexes][indexes2]
        N           = len(x_bin)
        x_mean[i]   = x_bin.mean()
        x_error[i]  = np.sqrt( ( np.sum( (x_bin - x_mean[i])**2 ) + (xerr**2)*N ) / (N*(N-1)) )

        ybin        = y[indexes][indexes2]
        y_total[i]   = ybin.sum(axis=0)
        #y_error[i]  = np.sqrt( np.sum( (ybin - y_mean[i]) **2 ) /  ( N - 1 ) ) / np.sqrt(N)

    # print len(x_mean), len(bin_widths)
    return x_mean, y_total, x_error#, np.array(bin_widths)#, y_error


def bin_data_fixed_bins(x, y, bin_width = 10, xerr=0, range=[], total=False):
    """ Bins two arrays based on the value of x. Splits x into several bins of size bin_width (same units as x), and
    returns the mean values of x and y in each bin. It also calculates and returns the standard error on y, due to binning.
    :return: x, y, y_err """
    #min_x, max_x = min(x), max(x)
    #number_means = int(np.ceil( (max_x - min_x)/bin_width ))     # Number of x-points (mean values stored in bins)

    if range == []:
        min_x, max_x = min(x), max(x)
        number_means = int(np.ceil( (max_x - min_x)/bin_width ))     # Number of x-points (mean values stored in bins)
        x_bins = np.linspace(min_x, max_x, number_means+1)
        x_mean, y_mean, x_error, y_error = np.zeros(number_means), np.zeros(number_means), np.zeros(number_means), np.zeros(number_means)

    else:
        #number_means = int(np.ceil( (range[1] - range[0])/bin_width ))     # Number of x-points (mean values stored in bins)
        x_bins = np.arange(range[0], range[1], bin_width)#, number_means+1)
        number_means = len(x_bins) - 1
        min_x, max_x = range[0], range[1]
        x_mean, y_mean, x_error, y_error = np.zeros(number_means), np.zeros(number_means), np.zeros(number_means), np.zeros(number_means)


    x_mean = ( x_bins[:-1] + x_bins[1:] ) / 2

    for i in xrange(0, len(x_bins)-1, 1):
        indexes     = np.where(x_bins[i] <= x)
        indexes2    = np.where(x[indexes] <= x_bins[i+1])

#        N           = len(x_bins[i])
#        x_error[i]  = np.sqrt( ( np.sum( (x_bin - x_mean[i])**2 ) + (xerr**2)*N ) / (N*(N-1)) )

        ybin        = y[indexes][indexes2]
        N = len(ybin)
        if total:
            y_mean[i] = ybin.sum(axis=0)
        else:
            y_mean[i]   = ybin.mean(axis=0)
            y_error[i]  = np.sqrt( np.sum( (ybin - y_mean[i]) **2 ) /  ( N - 1 ) ) / np.sqrt(N)



    return x_mean, y_mean, y_error


def bin_data_quick(x, y, bin_width, x_err=0):
    """
    A simpler and quicker version of bin_data_x_space
    """
    min_x, max_x = min(x), max(x)
    number_means = int(np.ceil( (max_x - min_x)/bin_width ))     # Number of x-points (mean values stored in bins)
    x_bins = np.linspace(min_x, max_x, number_means+1)
    indexes = np.digitize(x, bins=x_bins)
    x2 = [x[indexes == i].mean() for i in range(1, len(x_bins))]
    y2 = [y[indexes == i].mean() for i in range(1, len(x_bins))]
    # Create x error and y error
    yerr=[]
    xerr=[]
    for i in xrange(1, len(x_bins), 1):
        y_bin = y[indexes == i]
        yerr.append( ( ((y_bin - y2[i-1])**2).sum()/((len(y_bin)-1)*len(y_bin)) )**0.5 )
        x_bin = y[indexes == i]
        xerr.append(x_err/(len(x_bin))**0.5)
    return x2, y2, np.array(xerr), np.array(yerr)


def save_arrays(arrays, folder=".", save_names=""):
    """
    Save multiple arrays to a .npy file.
    :param arrays: a list of arrays
    :param folder: the folder in which to save the arrays
    :param save_names: a list of file names, with which to save the arrays
    """
    for i in xrange(0, len(arrays), 1):
        np.save(folder+"/"+save_names[i], arrays[i])


def load_arrays(file_names):
    """
    Load multiple arrays from multiple .npy files specified by file_names
    :param file_names: a list of file names, [file1.npy, ..., fileN.npy]
    :return: a list of the arrays. [array1.npy, ..., arrayN.npy]
    """
    arrays = []
    for i in xrange(0, len(arrays), 1):
        arrays.append(np.load(file_names[i]+".npy"))
    return arrays


def replace_all_with_ones(array):
    """ replace all non-zero entries in a 2D numpy array with ones. """    
    array[array.nonzero()] = 1
    return array


def scramble_array(grid):
    """ Randomise the positions of the entries in grid """
    for i in range(grid.shape[0]):      # Iterate over columns
        r = np.random.randint(0, grid.shape[0])
        grid[i,:] = np.roll(grid[i,:], r)

    for j in range(grid.shape[1]):
        r = np.random.randint(0, grid.shape[1])
        grid[:,j] = np.roll(grid[:,j], r)

    return grid


def eliminate_null_groups(arrays):
    """ Expects arrays = [groups, array1, ..., arrayN]. """
    assert type(arrays) == list
    indexes = np.where(arrays[0] != 1073741824)
    for i in xrange(0, len(arrays), 1):
        arrays[i] = arrays[i][indexes]
    return arrays

# Test code
if __name__ == '__main__':
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    centers, freqs = histogram(x, 3)