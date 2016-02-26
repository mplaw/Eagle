"""
    Contains code to calculate and plot histograms, column density distribution functions and the first moment
    of the column density distribution function.
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
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import colors
from PhysicalConstants import *
from SnapshotParameters import *
from MiscellaneousFunctions import *
from matplotlib.patches import Rectangle, Circle
import ProjectFiles
import scipy.optimize

__author__ = 'Matt'

# User defined information =========================================================================================== #
small_fig_width_pt = 223.51561 + 18+12.5        # Get this from LaTeX using \showthe\columnwidth
big_fig_width_pt = 526.72534 + 25 + 18*2

fig_width_pt    = big_fig_width_pt
inches_per_pt   = 1.0/72.27                     # Convert pt to inches
golden_mean     = 0.6                           # Aesthetic ratio
fig_width       = fig_width_pt*inches_per_pt    # width in inches
fig_height      = fig_width*golden_mean         # height in inches
fig_size        = [fig_width,fig_height]

params = {
        'backend'               : 'ps',
        'mathtext.default'      : 'regular',
        'font.family'           : 'Times',
        'font.size'             : 11,
        'font.weight'           : 400,          #'bold',
        'axes.labelsize'        : 11,
        'legend.fontsize'       : 8,
        'xtick.labelsize'       : 8,
        'ytick.labelsize'       : 8,
        'xtick.major.size'      : 1,            # major tick size in points
        'xtick.minor.size'      : .5,
        'ytick.major.size'      : 1,            # major tick size in points
        'ytick.minor.size'      : .5,
        'text.usetex'           : False,
        'text.latex.unicode'    : True,
        'figure.figsize'        : fig_size
        }
plt.rcParams.update(params)


# Calculated constant
Boxsizecgs = Snapshot.BoxsizepMpc * Constant.Mpc

# Importing data ===================================================================================================== #
save_folder   = "Histograms/"
#grid_name     = ProjectFiles.z3gridHI         # "3NGrids3RMin128RMax2048D0.1Grid.npy" # "3NGrids3RMin128RMax2048D0.001Grid.npy"

#cGrid = np.load(grid_name)

base_save_name = "test"

str_unit_con = "a"

# Observation data (Hardcoded) ======================================================================================= #
def observational_data():
    # z ~ 2.4
    #k02 - Lynman-alpha forest
    z_k02 = 2.34
    log_NI_bins_k02 = np.arange(12.50, 14.51, 0.50)
    f_k02 = np.array( [-11.19, -11.76, -12.54, -13.30] )
    #OPB07
    z_OPB07 = 2.51
    log_NI_bins_OPB07 = np.array( [19.00,19.60,20.30] )
    f_OPB07 = np.array( [-20.63, -21.50] )
    #PW09
    z_PW09 = 2.51
    log_NI_bins_PW09 = np.arange(20.30, 22.31, 0.2)
    f_PW09 = np.array( [-21.83, -22.29, -22.41, -22.88, -23.36, -23.60, -24.33, -25.00, -99.00, -99.00] )
    # z ~ 3.7
    #k01
    z_k01 = 3.75
    log_NI_bins_k01 = np.array( [13.65, 14.05] )
    f_k01 = np.array( [-12.42] )
    #OPB07
    z_OPB07_2 = 3.58
    log_NI_bins_OPB07_2 = np.array( [19.00,19.60,20.30] )
    f_OPB07_2 = np.array( [-20.37, -21.22] )
    #PW09

    # C. Peroux et al. http://mnras.oxfordjournals.org/content/363/2/479.abstract?sid=dfaf75fd-67a1-4acb-944f-43a4797979ba (only taken low redshift data)
    # http://mnras.oxfordjournals.org/content/363/2/479.full.pdf+html?sid=dfaf75fd-67a1-4acb-944f-43a4797979ba
    red_per          = [1.78, 3.50]
    log_NHI_bins_per = [19.00, 19.50, 20.30, 20.65, 21.00, 21.35, 21.70]
    log_NHI_cent_per = [19.23, 19.94, 20.45, 20.81, 21.08, 21.42]
    log_f_per        = np.array([-20.5, -21.2, -21.9, -22.5, -23.0, -23.8])
    log_f_per_min    = np.array([-20.4, -21.1, -21.9, -22.4, -22.9, -23.6])
    log_f_per_max    = np.array([-20.8, -21.4, -22.0, -22.6, -23.2, -24.1])
    log_f_per_u_error= log_f_per_max - log_f_per
    log_f_per_l_error= log_f_per - log_f_per_min
    x_error_per = [0.25/(6**0.5), 0.4/(9**0.5), 0.175/(33*0.5), 0.175/(20**0.5), 0.175/(13**0.5), 0.175/(5**0.5)]

    # Rudie et al (via Prochaska)
    # http://dro.dur.ac.uk/12759/1/12759.pdf
    red_rud = 2.37
    log_NHI_bins_rud = np.array([ 13.01,  13.11,  13.21,  13.32,  13.45,  13.59,  13.74,  13.90,  14.08,  14.28,  14.50,  14.73,  15.00,  15.28,  15.60,  15.94,  16.32,  16.73,  17.20])
    log_f_rud        = np.array([-11.33, -11.48, -11.65, -11.76, -11.96, -12.23, -12.53, -12.78, -13.11, -13.41, -13.74, -14.20, -14.75, -15.19, -15.66, -16.25, -16.91, -17.37])
    log_f_rud_u_error= np.array([  0.02,   0.02,   0.02,   0.02,   0.02,   0.03,   0.03,   0.03,   0.03,   0.04,   0.04,   0.05,   0.06,   0.07,   0.08,   0.09,   0.11,   0.11])
    log_f_rud_l_error= np.array([  0.02,   0.02,   0.02,   0.02,   0.03,   0.03,   0.03,   0.03,   0.04,   0.04,   0.04,   0.05,   0.07,   0.08,   0.09,   0.12,   0.16,   0.15])
    x_error_rud      = (log_NHI_bins_rud[1:] - log_NHI_bins_rud[:-1])/2 # Half the bin width

    # Noterdaemoe et al (via Prochaska)
    red_not = [2.3, 2.7]
    log_NHI_bins_not = np.array([ 20.30,  20.50,  20.70,  20.90,  21.10,  21.30,  21.50,  21.70, 21.90])
    log_f_not        = np.array([-21.80, -22.26, -22.38, -22.85, -23.33, -23.58, -24.30, -24.98])
    log_f_not_u_error= np.array([0.06, 0.08, 0.08, 0.11, 0.15, -0.16, 0.34, 0.76])
    log_f_not_l_error= np.array([0.06, 0.08, 0.07, 0.10, 0.15, -0.15, 0.30, 0.52])
    x_error_not = (log_NHI_bins_not[1:] - log_NHI_bins_not[:-1])/2
    print x_error_not


    # Find centers of data
    #log_NI_bin_centers_k02      = ( log_NI_bins_k02[:4] + log_NI_bins_k02[1:] ) / 2
    #log_NI_bin_centers_OPB07    = ( log_NI_bins_OPB07[0:2] + log_NI_bins_OPB07[1:3] ) / 2
    #log_NI_bin_centers_PW09     = ( log_NI_bins_PW09[:10] + log_NI_bins_PW09[1] ) / 2
    #log_NI_bin_centers_k01      = ( log_NI_bins_k01[0:1] + log_NI_bins_k01[1:2] ) / 2
    #log_NI_bin_centers_OPB07_2  = ( log_NI_bins_OPB07_2[0:2] + log_NI_bins_OPB07_2[1:3] ) / 2
    log_NI_bin_centers_rud      = ( log_NHI_bins_rud[1:] + log_NHI_bins_rud[:-1] ) / 2
    log_NI_bin_centers_not      = ( log_NHI_bins_not[1:] + log_NHI_bins_not[:-1] ) / 2

    # Extend all data into large lists
    centers = [log_NI_bin_centers_rud, log_NHI_cent_per,  log_NI_bin_centers_not]
    hist    = [log_f_rud, log_f_per, log_f_not]
    hist_min= [log_f_rud_l_error, log_f_per_u_error, log_f_not_l_error]
    hist_max= [log_f_rud_u_error, log_f_per_l_error, log_f_not_u_error]
    x_error = [x_error_rud, x_error_per, x_error_not]
    redshift= [red_rud,  red_per, red_not]
    name = ["Rudie et al. 2013", "Peroux et al. 2005", "Noterdaeme et al. 2009"]

    # Convert errros into   [ lower error, ... ]
    #                       [ upper error, ... ]
    hist_error = []
    for i in xrange(len(hist_min)):
        error = np.zeros((2, len(hist_min[i])))
        error[0,:] = hist_min[i]
        error[1,:] = hist_max[i]
        print error
        hist_error.append(error)
    print hist_error

    return centers, x_error, hist, hist_error, redshift, name


def altay_and_theuns():
    """
    Hard codes Altay and Theun's data (OWL's simulations) from the Through thick and thin pape
    :return: centre of bins, histogram, width of bin, redshift of data, name of data
    """
    name = "Owls"
    redshift = "3.00"
    #dz = 9.11e-3
    # Ray Traced :: 1000*X_1 = 133.1
    bins    = np.arange(12.50, 17.05, 0.25)         # Slightly over esitmate the end point to include it in the list
    bins2   = np.arange(17.00, 22.51, 0.10)
    bins    = np.concatenate((bins, bins2))
    bins    = 10**bins                              # Convert out of log_10(space)
    width   = (bins[1:] - bins[:-1])                # Width was constant in log space, now non-constant.
    center  = (bins[:-1] + bins[1:]) / 2

    hist     = [3598, 4062, 4135, 3651, 2918, 2144, 1362, 842, 466, 254, 145, 73, 49, 40, 25, 19, 11]
    absDist = 133.1#/1000
    hist = np.array(hist)/absDist

    # Projected :: 16,3842**2 * X_1 = 3.574e7
    hist2    = [858492, 747955, 658685, 582018, 468662, 431614, 406575, 387631, 374532, 374532, 359789, 350348, 342146, 334534, 329178, 324411,320648,318207,316232,314852,314504,314583,313942,315942,315942,315802,316330,316884,317336,317979,316526,317212, 314774, 309333, 302340, 291816, 275818, 254368, 228520, 198641, 167671, 135412, 103583, 76651, 54326, 37745, 25140, 16784, 10938, 6740, 3667, 1614, 637, 206, 33, 14, 7]
    absDist2 = 3.574*10**7#/ (16384**2)
    hist2 = np.array(hist2)/absDist2#/(16384**2)
    hist = np.concatenate((hist, hist2))

    # Correct hist to ccdf
    hist /= width

    return center, np.array(hist), width, redshift, name


# Histogram functions ================================================================================================ #


def absorption_distance():
    """ :return: The absorption distance """
    absDistdz = (Snapshot.H0/Constant.c)*(1+Snapshot.Redshift)**2 * Snapshot.BoxsizecMpc#*Constant.Mpc
    return absDistdz


def histogram(grid, remove_empty_bins=True, number_of_bins=500):
    """ Plot a histogram of column densities in grid. """
    #grid = ma.masked_where(grid == 0, grid)                            # Zero's can't be logged
    hist, bins2  = np.histogram(np.log10(grid), bins=number_of_bins)    # Create histogram frequencies and bins: note histogram bins linearly - so crap for many orders of mag
    bins = 10**bins2#/np.log(10)                                        # Convert bins, center and bins back to the unlogged space
    width       = 1 * (bins[1:] - bins[:-1])                            # Width was constant in log space, now non-constant.
    center      = (bins[:-1] + bins[1:]) / 2

    if remove_empty_bins:
        hist2 = []
        cent2 = []
        width2 = []
        for i, j in enumerate(hist):                                    # Remove boring bins with nothing in
            if j != 0:
                hist2.append(j)
                cent2.append(center[i])
                width2.append(width[i])
        center      = np.array(cent2)
        hist        = np.array(hist2)
        width       = np.array(width2)
    return center, hist, width


# Column density distribution function
def cddf(grid):
    """ Return the column density distribution function (cddf) of grid. """
    center, hist, bin_width = histogram(grid, remove_empty_bins=False)
    hist = hist/bin_width
    dz          = 1 # Snapshot.Hz * Boxsizecgs / Constant.c
    nPixels     = grid.shape[0]*grid.shape[1]
    absDistdz   = absorption_distance()
    hist        /= (absDistdz*1*nPixels*dz)
    return center, hist


def cddf_error(log10_space=True):
    """ Return the error in the column density distribution function. """
    grid_front = np.load(ProjectFiles.z3gridHIhalf1)
    grid_back = np.load(ProjectFiles.z3gridHIhalf2)
    center_f, hist_f = cddf(grid_front)
    center_b, hist_b = cddf(grid_back)

    # in log-space
    if log10_space:
        hist_b = ma.masked_where(hist_b == 0, hist_b)
        hist_f = np.log10(hist_f)
        hist_b = np.log10(hist_b)
    hist_error = np.sqrt( (np.sum((hist_f-hist_b)**2)/len(hist_f) ) )
    return hist_error


def plot_ccdf(grid, plot_observational_data=True, save_name="Histograms/z3Histogram.pdf", fit_power_law=False):
    """ Plot the column density distribution function (cddf) of grid. """
    hist_err = cddf_error()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # Plot Eagle with errors
    colour = 'b'

    center, hist = cddf(grid)
    x = np.log10(center)
    y = np.log10(hist)
    y_err = hist_err

    # Save results
    np.save("Histograms/z2center",center)
    np.save("Histograms/z2hist", hist)
    np.save("Histograms/z2hist_err", hist_err)

    plt.plot(x, y, color=colour, label='z=3.00 Eagle', linewidth=2)
    ax.fill_between (x, y-y_err, y+y_err, alpha=0.2, antialiased=True, linestyle='dashed', facecolor=colour, edgecolor=colour)
    #ax.fill_betweenx(y, x-x_err, x+x_err, alpha=0.2, antialiased=True, facecolor=colour, edgecolor=colour)                # facecolor='#F0F8FF', alpha=1.0, edgecolor='#8F94CC', linewidth=1, linestyle='dashed')

    # Plot a power law
    if fit_power_law:
        hist, center = eliminate_zeros(hist, [center])
        center=center[0]
        new_parameters = minimise_chi_squared(center, hist, hist_err)
        model = power_law(new_parameters, center)

        plt.plot(np.log10(center), np.log10(model), color='m', label="z=3.00 Eagle power law", alpha=0.80)


    # Plot observational data
    if plot_observational_data:
        c, h, width, red, name = altay_and_theuns()
        #h = h*(1/((10**c))*(np.log(10)/2))

        plt.plot(np.log10(c), np.log10(h), color='r', label="z="+str(red)+" "+name)
        colours = ['g','black','grey','y','m','c']
        c, c_error, h, hist_error, redshift, name = observational_data()
        for i in xrange(0, len(c), 1):
            #print hist_error[i][0][0]
            plt.errorbar(c[i],h[i], xerr=c_error[i], yerr=hist_error[i],color=colours[i%len(colours)], label="z="+str(redshift[i])+" "+name[i], linestyle='none', elinewidth=2)
        plt.legend(frameon=False, loc=3)

    # Cosmetics ====================================================================================================== #
    # Fill regions
    x_min = 13
    x_lls = 17.2
    x_dla = 20.3
    x_max = 23
    y_min = -30
    y_max = -10
    fill_y = [-10,-30]
    fill_x1 = [12, 17.2]
    alpha_rect = plt.Rectangle((x_min, y_min), x_lls - x_min, y_max-y_min, facecolor="blue", edgecolor='none', alpha=0.05)
    fig.gca().add_artist(alpha_rect)
    lls_rect = plt.Rectangle((x_lls, y_min), x_dla - x_lls, y_max-y_min, facecolor="green", edgecolor='none', alpha=0.05)
    fig.gca().add_artist(lls_rect)
    dla_rect = plt.Rectangle((x_dla, y_min), x_max - x_dla, y_max-y_min, facecolor="red", edgecolor='none', alpha=0.05)
    fig.gca().add_artist(dla_rect)

    # Add lines to the region boundaries
    plt.axvline(x_lls,color='green', linestyle='--', alpha=0.2)
    plt.axvline(x_dla,color='red', linestyle='--', alpha=0.2)

    # Add minor ticks and inrease the size of all ticks
    ax.minorticks_on()
    ax.tick_params('both', length=6, width=1, which='major')
    ax.tick_params('both', length=3 , width=1, which='minor')

    plt.xlim((x_min, x_max))
    plt.ylim((-30, -10))
    plt.xlabel(r"$\rm log_{10} \ (N_{HI}/cm^{-2})$", labelpad=3)                # Column Density bins
    plt.ylabel(r"$\rm log_{10} \ (f \ (N_{HI},z)/cm^{2}) $", labelpad=2)        # Column Density Frequency
    plt.savefig(save_name, dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.show()
    plt.close()
    return center, hist, hist_err


def first_moment_cddf_error(log10_space=True):
    """ The first moment of the column density distribution function (cddf) of grid. """
    grid_front = np.load(ProjectFiles.z3gridHIhalf1)
    grid_back = np.load(ProjectFiles.z3gridHIhalf2)
    center_f, hist_f = cddf(grid_front)
    center_b, hist_b = cddf(grid_back)

    hist_b *= center_b
    hist_f *= center_f

    # in log-space
    if log10_space:
        hist_b = ma.masked_where(hist_b == 0, hist_b)
        hist_f = np.log10(hist_f)
        hist_b = np.log10(hist_b)
    hist_error = np.sqrt( (np.sum((hist_f-hist_b)**2)/len(hist_f) ) )
    #print "Error on histogram: ", hist_error
    return hist_error


def plot_first_moment_of_ccdf(grid, plot_observational_data=True, save_name="Histograms/z3firstmomentHistogram2.pdf"):
    """ Plot the first moment of the column density distribution function (cddf) of grid. """
    hist_err = first_moment_cddf_error()
    #print "Histogram error", hist_err

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #plt.loglog()
    #plt.plot(center, hist)

    # Plot Eagle with errors
    colour = 'b'
    center, hist = cddf(grid)
    hist2 = hist*center
    x = np.log10(center)
    y = np.log10(hist2)
    y_err = hist_err

    # Save results
    #np.save("Histograms/z2center",center)
    #np.save("Histograms/z2hist", hist)
    #np.save("Histograms/z2hist_err", hist_err)

    #print "Hist: ", hist
    #print "Hist-error: ", hist - hist_err
    #print "log10(hist-error): ", np.log10(hist - hist_err)
    plt.plot(x, y, color=colour, label='z=3.00 Eagle', linewidth=2)
    ax.fill_between (x, y-y_err, y+y_err, alpha=0.2, antialiased=True, linestyle='dashed', facecolor=colour, edgecolor=colour)
    #ax.fill_betweenx(y, x-x_err, x+x_err, alpha=0.2, antialiased=True, facecolor=colour, edgecolor=colour)                # facecolor='#F0F8FF', alpha=1.0, edgecolor='#8F94CC', linewidth=1, linestyle='dashed')

    # Plot a power law
    #hist, center = eliminate_zeros(hist, [center])
    #center=center[0]
    #new_parameters = minimise_chi_squared(center, hist, hist_err)
    #model = power_law(new_parameters, center)
    #plt.plot(np.log10(center), np.log10(model), color='m', label="z=3.00 Eagle power law", alpha=0.80)

    # Plot observational data
    if plot_observational_data:
        c, h, width, red, name = altay_and_theuns()
        #h = h*(1/((10**c))*(np.log(10)/2))
        plt.plot(np.log10(c), np.log10(h*c), color='r', label="z="+str(red)+" "+name)
        colours = ['g','black','grey','y','m','c']
        c, c_error, h, hist_error, redshift, name = observational_data()
        for i in xrange(0, len(c), 1):
            c_obs = c[i]
            h_obs = np.log10(10**np.array(h[i])*10**np.array(c[i]))
            c_obs_err = c_error[i]
            h_obs_err = h_obs[i] * (hist_error[i]/h[i])
            plt.errorbar(c_obs, h_obs, xerr=c_obs_err, yerr=h_obs_err, color=colours[i%len(colours)], label="z="+str(redshift[i])+" "+name[i], linestyle='none', elinewidth=2)
        plt.legend(frameon=False, loc=3)

    # Cosmetics ====================================================================================================== #
    # Fill regions
    x_min = 13#12
    x_lls = 17.2
    x_dla = 20.3
    x_max = 23
    y_min = -6
    y_max = 2
    fill_y = [-10,-30]
    fill_x1 = [12, 17.2]
    alpha_rect = plt.Rectangle((x_min, y_min), x_lls - x_min, y_max-y_min, facecolor="blue", edgecolor='none', alpha=0.05)
    fig.gca().add_artist(alpha_rect)
    lls_rect = plt.Rectangle((x_lls, y_min), x_dla - x_lls, y_max-y_min, facecolor="green", edgecolor='none', alpha=0.05)
    fig.gca().add_artist(lls_rect)
    dla_rect = plt.Rectangle((x_dla, y_min), x_max - x_dla, y_max-y_min, facecolor="red", edgecolor='none', alpha=0.05)
    fig.gca().add_artist(dla_rect)

    # Add lines to the region boundaries
    plt.axvline(x_lls,color='green', linestyle='--', alpha=0.2)
    plt.axvline(x_dla,color='red', linestyle='--', alpha=0.2)

    # Add minor ticks and inrease the size of all ticks
    ax.minorticks_on()
    ax.tick_params('both', length=6, width=1, which='major')
    ax.tick_params('both', length=3 , width=1, which='minor')

    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    plt.xlabel(r"$\rm log_{10} \ (N_{HI}/cm^{-2})$", labelpad=3)                # Column Density bins
    plt.ylabel(r"$\rm log_{10} \ (N_{HI}f \ (N_{HI},z)) $", labelpad=2)        # Column Density Frequency
    plt.savefig(save_name, dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.show()
    plt.close()
    return center, hist, hist_err


def plot_both(plot_observational_data=True, save_name="Histograms/z3Both.pdf"):
    """ Plot both the cddf and the first moment of the cddf. """
    fig = plt.figure()
    # CDDF
    ax1 = fig.add_subplot(1,2,1)
    hist_err = cddf_error()

    # Plot Eagle with errors
    colour = 'b'
    center, hist = cddf(grid)
    x = np.log10(center)
    y = np.log10(hist)
    y_err = hist_err

    ax1.plot(x, y, color=colour, label='z=3.00 Eagle', linewidth=2)
    ax1.fill_between (x, y-y_err, y+y_err, alpha=0.2, antialiased=True, linestyle='dashed', facecolor=colour, edgecolor=colour)

    # Plot a power law
    hist, center = eliminate_zeros(hist, [center])
    center=center[0]
    new_parameters = minimise_chi_squared(center, hist, hist_err)
    model = power_law(new_parameters, center)

    # Plot observational data
    if plot_observational_data:
        c, h, width, red, name = altay_and_theuns()
        #h = h*(1/((10**c))*(np.log(10)/2))

        ax1.plot(np.log10(c), np.log10(h), color='r', label="z="+str(red)+" "+name)
        colours = ['g','black','grey','y','m','c']
        c, c_error, h, hist_error, redshift, name = observational_data()
        for i in xrange(0, len(c), 1):
            #print hist_error[i][0][0]
            ax1.errorbar(c[i],h[i], xerr=c_error[i], yerr=hist_error[i],color=colours[i%len(colours)], label="z="+str(redshift[i])+" "+name[i], linestyle='none', elinewidth=2)
        #ax1.legend(frameon=False, loc=3)

    # Cosmetics ====================================================================================================== #
    # Fill regions
    x_min = 13
    x_lls = 17.2
    x_dla = 20.3
    x_max = 23
    y_min = -30
    y_max = -10
    fill_y = [-10,-30]
    fill_x1 = [12, 17.2]
    alpha_rect = plt.Rectangle((x_min, y_min), x_lls - x_min, y_max-y_min, facecolor="blue", edgecolor='none', alpha=0.05)
    fig.gca().add_artist(alpha_rect)
    lls_rect = plt.Rectangle((x_lls, y_min), x_dla - x_lls, y_max-y_min, facecolor="green", edgecolor='none', alpha=0.05)
    fig.gca().add_artist(lls_rect)
    dla_rect = plt.Rectangle((x_dla, y_min), x_max - x_dla, y_max-y_min, facecolor="red", edgecolor='none', alpha=0.05)
    fig.gca().add_artist(dla_rect)

    # Add lines to the region boundaries
    ax1.axvline(x_lls,color='green', linestyle='--', alpha=0.2)
    ax1.axvline(x_dla,color='red', linestyle='--', alpha=0.2)

    # Add minor ticks and inrease the size of all ticks
    ax1.minorticks_on()
    ax1.tick_params('both', length=6, width=1, which='major')
    ax1.tick_params('both', length=3 , width=1, which='minor')
    ax1.set_xlim((x_min, x_max))
    ax1.set_ylim((-30, -10))
    ax1.set_xlabel(r"$\rm log_{10} \ (N_{HI}/cm^{-2})$", labelpad=3)                # Column Density bins
    ax1.set_ylabel(r"$\rm log_{10} \ (f \ (N_{HI},z)/cm^{2}) $", labelpad=2)        # Column Density Frequency

    # 1st moment of CDDF
    ax2 = fig.add_subplot(1,2,2)

    hist_err = first_moment_cddf_error()
    # Plot Eagle with errors
    colour = 'b'
    center, hist = cddf(grid)
    hist2 = hist*center
    x = np.log10(center)
    y = np.log10(hist2)
    y_err = hist_err
    ax2.plot(x, y, color=colour, label='z=3.00 Eagle', linewidth=2)
    ax2.fill_between (x, y-y_err, y+y_err, alpha=0.2, antialiased=True, linestyle='dashed', facecolor=colour, edgecolor=colour)

    # Plot observational data
    if plot_observational_data:
        c, h, width, red, name = altay_and_theuns()
        #h = h*(1/((10**c))*(np.log(10)/2))

        ax2.plot(np.log10(c), np.log10(h*c), color='r', label="z="+str(red)+" "+name)
        colours = ['g','black','grey','y','m','c']
        c, c_error, h, hist_error, redshift, name = observational_data()
        for i in xrange(0, len(c), 1):
            c_obs = c[i]
            h_obs = np.log10(10**np.array(h[i])*10**np.array(c[i]))
            c_obs_err = c_error[i]
            h_obs_err = h_obs[i] * (hist_error[i]/h[i])
            ax2.errorbar(c_obs, h_obs, xerr=c_obs_err, yerr=h_obs_err, color=colours[i%len(colours)], label="z="+str(redshift[i])+" "+name[i], linestyle='none', elinewidth=2)
        ax2.legend(frameon=False, loc=3)

    # Cosmetics ====================================================================================================== #
    # Fill regions
    x_min = 13#12
    x_lls = 17.2
    x_dla = 20.3
    x_max = 23
    y_min = -6
    y_max = 2
    fill_y = [-10,-30]
    fill_x1 = [12, 17.2]
    alpha_rect = plt.Rectangle((x_min, y_min), x_lls - x_min, y_max-y_min, facecolor="blue", edgecolor='none', alpha=0.05)
    fig.gca().add_artist(alpha_rect)
    lls_rect = plt.Rectangle((x_lls, y_min), x_dla - x_lls, y_max-y_min, facecolor="green", edgecolor='none', alpha=0.05)
    fig.gca().add_artist(lls_rect)
    dla_rect = plt.Rectangle((x_dla, y_min), x_max - x_dla, y_max-y_min, facecolor="red", edgecolor='none', alpha=0.05)
    fig.gca().add_artist(dla_rect)

    # Add lines to the region boundaries
    ax2.axvline(x_lls,color='green', linestyle='--', alpha=0.2)
    ax2.axvline(x_dla,color='red', linestyle='--', alpha=0.2)

    # Add minor ticks and inrease the size of all ticks
    ax2.minorticks_on()
    ax2.tick_params('both', length=6, width=1, which='major')
    ax2.tick_params('both', length=3 , width=1, which='minor')

    ax2.set_xlim((x_min, x_max))
    ax2.set_ylim((y_min, y_max))
    ax2.set_xlabel(r"$\rm log_{10} \ (N_{HI}/cm^{-2})$", labelpad=3)                # Column Density bins
    ax2.set_ylabel(r"$\rm log_{10} \ (N_{HI}f \ (N_{HI},z)) $", labelpad=2)        # Column Density Frequency
    plt.savefig(save_name, dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.show()

    plt.show()



# Fit power law ====================================================================================================== #


def eliminate_zeros(x, others):
    """ Eliminate elements for both x and others (list of arrays), for which the corresponding element in x is 0. """
    indexes = np.where(x != 0)
    for i in xrange(len(others)):
        others[i] = np.array(others[i])[indexes]
    x = x[indexes]
    return x, others


def power_law(parameters, x):
    """ Returns a power law: parameters[0]*(x**parameters[1]). """
    return parameters[0]*(x**parameters[1])


def power_law_ln(parameters, x):
    """ Returns a log of a power law: parameters[0] - parameters[1]*x. """
    return parameters[0] + parameters[1]*x


def residual(parameters, x, y, y_error):
    """ Return residuals (y - model)/error.  """
    model = power_law_ln(parameters, x)
    return (y-model)/y_error


def chi_squared(parameters, x, y, y_error):
    """ Return the reduced chi-squared between a model and data. """
    model = power_law_ln(parameters, x)
    return np.sum(((y-model)/y_error)**2)/len(y)


def minimise_chi_squared(x, y, y_error):
    """ Change the parameters of a power law to minimised the reduced chi-squared.
        Note: optimise functions work better in log-space, so everything is converted to log-space first and
        then converted back after fitting.
        Note2: Optimise also works terribly with very small numbers, so upscale everything and then convert back."""

    # Initialise parameters
    parameters = [1000000.2377, -1.4]
    print "Initial guess: ", parameters

    # Convert to log-space
    #y = ma.masked_where(y==0, y)
    log_x = np.log10(x)
    log_y = np.log10(y)
    log_y_error = y_error#/y

    results = scipy.optimize.leastsq(residual, x0=parameters, args=(log_x, log_y, log_y_error), full_output=1)
    log_parameters = results[0]
    covar = results[1]

    print "Fitted log_10 parameters: ", log_parameters
    new_parameters = [10**log_parameters[0], log_parameters[1]]

    indexErr = np.sqrt( covar[0][0] )
    ampErr = np.sqrt( covar[1][1] ) * new_parameters[0]
    print "Fitted parameters: a ", new_parameters[0], " +- ", ampErr
    print "                   b ", new_parameters[1], " +- ", indexErr
    print "Minimised reduced chi-squared: ", chi_squared(log_parameters, log_x, log_y, log_y_error)
    return new_parameters


def load_hist():
    center = np.load("Histograms/z3center.npy")
    hist = np.load("Histograms/z3hist.npy")
    hist_err = np.load("Histograms/z3hist_err.npy")
    return center, hist, hist_err


# Mass density ======================================================================================================= #


def integrated_power_law(x, a, b):
    return (a/(2-b)) * x**(2-b)


def mass_density_neutral(x, p):
    """ Calculates the mass density of neutral Hydrogen by integrating the CDDF in power law form.
        see http://www.ast.cam.ac.uk/~pettini/Physical%20Cosmology/lecture12.pdf """
    # Y is the baryonic mass fraction
    Y = 0.752
    return (1/Snapshot.p_c)*(Snapshot.H0/Constant.c)*(Constant.m_HI/(1-Y)) * integrated_power_law(x, p[0], p[1])


# Run code =========================================================================================================== #
if __name__ == '__main__':
    # Load grid
    Snapshot.importSnap(ProjectFiles.z3Snapshot, ProjectFiles.z3Urchin)
    #grid = np.load(ProjectFiles.z3gridHI)
    grid = np.load(ProjectFiles.z3gridHI)

    print "Minimum and maximum N_HI", np.min(grid), np.max(grid)
    plot_both()
    center, hist, hist_err = plot_first_moment_of_ccdf(grid,save_name="Histograms/z3FirstMomentHistogram3.pdf")
    center, hist, hist_err = plot_ccdf(grid,save_name="Histograms/z3Histogram.pdf")


    # -- OR ---
    #center, hist, hist_err = load_hist()

    # Fit a power law
    hist, center = eliminate_zeros(hist, [center])
    center=center[0]
    new_parameters = minimise_chi_squared(center, hist, hist_err)
    model = power_law(new_parameters, center)

    plt.figure()
    plt.errorbar(np.log10(center), np.log10(hist), hist_err)
    plt.plot(np.log10(center), np.log10(model), color='r', label="Scipy Leastsq")
    plt.legend(frameon=False)
    plt.show()

    # Integrate the power-law to find the neutral mass density
    plt.figure()
    plt.plot(np.log10(center), np.log10(mass_density_neutral(center, new_parameters)))
    plt.show()