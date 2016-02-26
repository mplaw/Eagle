""" This file can auto-correlate or cross-correlate 2D arrays, and plot the results.
    Code: Python 2.7.9
    Requirements: SnapshotParameters, MiscellaneousFunctions, Numpy, Scipy, Matplotlib
    Functions:
        fourier_transform(array)                            :   Return the normalised Fourier transform of 'array'.
        inverse_fourier_transform(array)                    :   Return the normalised inverse Fourier transform of 'array'.
        power_spectrum(array)                               :   Return the normalised power spectrum of 'array'.
        auto_correlation_two_point(grid)                    :   Directly compute (slow!) the 2-point auto-correlation function of the 2D array, 'grid'.
        auto_correlation_two_point_fourier(grid)            :   Use Fourier transforms to compute the auto-correlation.
        cross_correlation_two_point_fourier(grid1, grid2)   :   Use Fourier transforms to compute the cross-correlation.
    To use:
        Change all parameters under "SET ME !!!" heading to ones specific to your system.
        Run this file.
"""
from __future__ import division
import numpy as np
import scipy
from scipy import fftpack
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy.ma as ma
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
from SnapshotParameters import *            # Gives me access to the Snapshot object (all snapshots related parameters)
from MiscellaneousFunctions import *

__author__ = 'Matt Law'

# SET ME !!! ********************************************************************************************************* #
grid_name = "grid1.npy"     # Full name (directory + file name + file extension) of the grid produced by "CreateGrid.py"
save_name = "Correlate1"    # Name to save graphs by.


# FUNCTIONS ********************************************************************************************************** #
# Fourier functions ================================================================================================== #
def fourier_transform(array):
    """ Returns the normalised discrete Fourier transform of 'array'.
    :param array: 1D or 2D numpy array or list (which will be converted to a numpy array)
    :return: F(array)/sqrt(N), where N is the length of the array. """
    if type(array) != np.ndarray:
        array = np.array(array)

    if array.ndim == 1:
        return fftpack.fft(array) / np.sqrt(len(array))
    elif array.ndim == 2:
        return fftpack.fft2(array) / np.sqrt(array.shape[0]*array.shape[1])
    else:
        print "Cannot Fourier transform array with dimension ", array.ndim


def inverse_fourier_transform(array):
    """ Returns the normalised inverse discrete Fourier transform of 'array'.
    :param array: 1D or 2D numpy array or list (which will be converted to a numpy array)
    :return: F^-1(array)*sqrt(N), where N is the length of the array."""
    if type(array) != np.ndarray:
        array = np.array(array)

    if array.ndim == 1:
        return fftpack.ifft(array) * np.sqrt(len(array))
    elif array.ndim == 2:
        return fftpack.ifft2(array) * np.sqrt(array.shape[0]*array.shape[1])
    else:
        print "Cannot inverse Fourier transform array with dimension ", array.ndim


def power_spectrum(array):
    """ Return the NORMALISED power spectrum of a 1-D or 2-D array. """
    if type(array) != np.ndarray:
        array = np.array(array)

    if array.ndim == 1:
        return np.abs(fftpack.fft(array))**2 / (len(array))
    elif array.ndim == 2:
        return np.abs(fftpack.fft2(array))**2 / (array.shape[0]*array.shape[1])
    else:
        print "Cannot Fourier transform array with dimension ", array.ndim


def auto_correlation_two_point(grid):
    """Finds the auto-correlation function, E(d), directly from P(d) = < p(r)p(r+d) > = mean_density**2 ( 1 + E(d) )
    :param grid: A 2D array of 0's and values. """
    mean_density = np.sum(grid)/(grid.shape[0]*grid.shape[1]) # Mean Column density of a single cell || units of [#/cm^2]
    # 0 <= delta <= size of grid/2 (assume grid is square)
    delta = np.arange(0, int(grid.shape[0]/2), 1)
    E = np.zeros(len(delta)*len(delta))
    mod_delta =  np.zeros((len(delta), len(delta)))
    for k in delta:
        for l in delta:
            E[k*int(grid.shape[0]/2)+l] = np.mean(grid*np.roll(np.roll(grid, k, axis=0), l, axis=1))    # Mean finds the average (a.k.a <...> ), and has replaced: sum instead of mean with an extra 1/N
            mod_delta[k, l] = np.sqrt(k*k + l*l)
    mod_delta = mod_delta.flatten()
    E = E/(mean_density**2) - 1
    mod_delta, E = sort2(mod_delta, E)                      # Sort for prettier plotting
    return mod_delta, E


def auto_correlation_two_point_fourier(grid):
    """
    Uses the relation that P(d) = <p(r)p(r+d)> = ... = (1/N)*F[|p(k)|**2](k)
                                Rough derivation
    Start with Probability of pair, P(d) = <p(r)p(r+d)>
    Replace p(r) and p(r+d) with their inverse fourier transforms (discrete):
        p(r)   = (1/sqrt(N) * sum [ DFT[ p(r) ](k) * exp(2 * pi * i * k   *   n / N) ]
        p(r+d) = (1/sqrt(N) * sum [ DFT[p(r+d)](q) * exp(2 * pi * i * q * (n+d) / N) ]
    P(d) = < 1/N sum sum DFT[p(r)]*DFT[p(r+d)]*exp(2*pi*i*k*n/N)*exp(2*pi*i*q*(n+d)/N) >
    Given that p(r+d) is real (so is p(r)) it can be replaced by it's complex conjugate
    P(d) = < 1/N sum sum DFT[p(r)]*DFT[p(r+d)]* *exp(2*pi*i*(k-q)*n/N)*exp(-2*pi*i*q*d/N) >
    When the average over all r, < .. > is taken the exp(2*pi*i*(k-q)*n/N) is just a plane wave and so averages to 0, unless k = q
    P(d) = 1/N sum ( |p(k)|**2 * exp(-2 * pi * i * q * d / N) = 1/sqrt(N) * DFT[|p(k)|**2]
    Then use E = P / (mean_density**2) - 1
    :param grid: A 2-D numpy array
    :return: |d|, E(d)
    """

    N = grid.shape[0]*grid.shape[1]                     # Normalisation factor, with which to divide the fourier transforms by

    grid = replace_all_with_ones(np.copy(grid))         # Make grid only 0s and 1s

    mean_density = np.mean(grid)#np.sum(grid)/(N)       # Mean Column density of a single cell || units of [#/cm^2]
    # Find auto-correlation function, E
    Power = power_spectrum(grid)
    #del grid
    Prob  = fourier_transform(Power)/np.sqrt(N)         # Extra 1/sqrt(N) is due to the left overs from re-absorbing terms into a DFT
    del Power
    Prob  = np.abs(Prob)                                # Convert to real numbers
    E     = Prob/(mean_density**2) - 1                  # Find the auto-correlation function, E(d): P = mean_density**2 * (1 + E(d))
    del Prob
    # Construct |delta|                                 # 0 <= delta <= grid.shape[0]/2
    row, col = np.indices((grid.shape[0], grid.shape[1]))
    mod_delta = (row**2 + col**2)**0.5
    del row, col

    E = E.flatten()
    mod_delta = mod_delta.flatten()
    mod_delta, E = sort2(mod_delta, E)                  # Sort for prettier plotting
    # Chop down data to half-size to prevent repeating information
    idxs = np.where( mod_delta < max(mod_delta)/2 )
    mod_delta = mod_delta[idxs]
    E = E[idxs]
    return mod_delta, E


def cross_correlation_two_point_fourier(grid1, grid2):
    """
    Cross-correlate 2 grids (2D arrays), using fourier transforms.
    :param grid1: 2D numpy array
    :param grid2: 2D numpy array
    :return: The radius of separation (d) , The cross-correlation function (Xi(d))
    DERIVATION :: http://mathworld.wolfram.com/Cross-CorrelationTheorem.html
        <p(r)n(r+d)> = 1/sqrt(N) * DFT[p(k)n(q)*],
    because (write in terms of inverse discrete fourier transforms):
        p(r)   = 1/sqrt(N) sum [ p(k) e^ikr ]
        n(r+d) = 1/sqrt(N) sum [ n(q) e^iq(r+d) ]
    real density functions, so are equal to their complex conjugates. Just conugating n:
        n(r+d) = (n(r+d))* = 1/sqrt(N) sum [ n(q)*  e^-iq(r+d) ]
    therefore:
        <p(r)n(r+d)> = < 1/N sum sum [ p(k)n(q)*  e^i*(k-q)r * e^-iqd ] >
    we know that e^i*(k-q)r is just a plane wave, which transforms to a dirac-delta function: 1 when k = q.
        <p(r)n(r+d)> =  1/N sum [ p(k)n(k)*  e^-ikd ]
        <p(r)n(r+d)> =  1/sqrt(N) DFT [ p(k)n(k)* ]
    """

    assert grid1.shape[0] == grid2.shape[0]
    assert grid1.shape[1] == grid2.shape[1]

    # Normalisation coefficient for Xi (=E), see doc string for derivation.
    Nx= grid1.shape[0]
    Ny= grid1.shape[1]
    N = Nx*Ny

    # Make grids 0s and 1s
    grid1 = replace_all_with_ones(np.copy(grid1))
    grid2 = replace_all_with_ones(np.copy(grid2))

    # Mean Column density of a single cell || units of [#/cm^2]
    mean_density1 = np.mean(grid1)
    mean_density2 = np.mean(grid2)

    # Take normalised Fourier transforms
    F1 = fourier_transform(grid1)
    F2 = fourier_transform(grid2)
    F2 = F2.conjugate()
    F1 = F1*F2
    del F2

    # Calculate the correlation function                  P = mean_density^2 * ( 1 + Xi(r) )
    Prob  = fourier_transform(F1)/np.sqrt(N)             # Extra 1/sqrt(N) due to DFT's - see derivation.
    del F1
    Prob  = np.abs(Prob)
    E     = Prob/(mean_density1*mean_density2) - 1
    del Prob

    # Convert to radial space :: construct |delta|
    row, col = np.indices((grid1.shape[0], grid1.shape[1]))
    mod_delta = (row**2 + col**2)**0.5
    del row, col
    E = E.flatten()
    mod_delta = mod_delta.flatten()
    mod_delta, E = sort2(mod_delta, E)                  # Sort for prettier plotting

    # Chop down data to half-size to prevent repeating information
    idxs = np.where( mod_delta < max(mod_delta)/2 )
    mod_delta = mod_delta[idxs]
    E = E[idxs]
    return mod_delta, E


# Plotting functions ================================================================================================= #
def plot_correlation(x, y, horizontal_line=None, bin=True, bin_size=1, units='pkpc', N=4096,
                     log_scale_x=True, log_scale_y=False, x_limit=None, y_limit=None, save_name="Correlation",
                     sampling_error=3.2, compare_to_zero=False):

    cell_to_pMpch = Snapshot.BoxsizecMpch / N
    cell_to_cMpc = Snapshot.BoxsizecMpc / N
    cell_to_pMpc = Snapshot.BoxsizepMpc / N
    cell_to_pkpc = Snapshot.Boxsizepkpc / N

    if units == 'cMpch':
        x = x*cell_to_pMpch
        x_units = r"$ \rm cMpch$"
    elif units =='pMpc':
        x = x*cell_to_pMpc
        x_units = r"$ \rm pMpc$"
    elif units =='cMpc':
        x = x*cell_to_cMpc
        x_units = 'cMpc'
    elif units == 'pkpc':
        x = x*cell_to_pkpc
        x_units = r"$ \rm pkpc$"
    else:
        x_units = 'cell number'

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Bin data
    if bin:
        x, y, x_err, y_err = bin_data_quick(x, y, bin_size, np.sqrt(2)/2)
    # Add sampling error to y_error
    y_err += sampling_error

    # Set labels and log scale
    ax.set_xlabel(r"$ \rm |\Delta| $"+"$ \ ($"+x_units+"$)$", labelpad=3)
    ax.set_ylabel(r"$ \rm \xi (\Delta) $", labelpad=2)
    if log_scale_x:
        ax.semilogx(nonposx='clip')
        ax.set_xlabel(r"$ \rm log_{10} \ |\Delta| $"+"$ \ ($"+x_units+"$)$", labelpad=3)
    if log_scale_y:
        ax.set_ylabel(r"$ \rm log_{10} \ \xi (\Delta) $", labelpad=2)
        ax.semilogy(nonposy='clip')

    ax.xaxis.set_major_formatter(ScalarFormatter())             # Set scalar numbers on tick labels, not 10^X
    ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.minorticks_on()                                          # Increase size of ticks
    ax.tick_params('both', length=6, width=1, which='major')
    ax.tick_params('both', length=3 , width=1, which='minor')

    # Plot the data
    ax.plot(x, y, color='b',linewidth=2, marker='.')

    # Plot the lower and upper limits
    ax.fill_between (x, y-y_err, y+y_err, alpha=0.2, antialiased=True, linestyle='dashed')
    ax.fill_betweenx(y, x-x_err, x+x_err, alpha=0.2)

    # Set axis limits
    if x_limit is not None:
        ax.set_xlim((0, x_limit))
    else:
        ax.set_xlim((0, max(x)))
    if y_limit is not None:
        ax.set_ylim((0, y_limit))
    else:
        if log_scale_y:
            ax.set_ylim((0.1, max(y)))
        else:
            ax.set_ylim((-1, max(y)))

    if compare_to_zero:
        indexes = np.where(x > 100)
        ax.set_xlim((100, max(x)))
        ax.set_ylim((min(y[indexes]), max(y[indexes])))

    # Plor a horizontal line at y=0 and others if pr
    plt.axhline(y=0, color='black', linestyle='--')

    if horizontal_line is not None:
        plt.axhline(y=horizontal_line, color='r')

    plt.savefig(save_name+".pdf", dpi=1000, bbox_inches='tight', pad_inches=0.04)
    plt.savefig(save_name+".png", dpi=300 , bbox_inches='tight', pad_inches=0.04)
    plt.savefig(save_name+".jpg", dpi=300 , bbox_inches='tight', pad_inches=0.04)
    plt.show()


def plot_correlations(xs, ys, horizontal_line=None, bin=True, bin_size=20, units='pkpc', N=4096,
                      log_scale_x=True, log_scale_y=False, x_limit=None, y_limit=None, save_name="Correlation",
                      sampling_error=3.2, labels=['LLS-LLS', 'GAL-GAL', 'LLS-GAL'], legend=False):
    """ Plots multiple correlations onto one graph and colours them differently.
    :param xs: [delta1, delta2, ... , deltaN]
    :param ys: [Xi1, Xi2, ... , XiN]
    :param horizontal_line: Plot additional horizontal lines at horizontal_line=location on y-axis
    :param bin: Bin the data?
    :param bin_size: Bin the data into equal-sized bins of bin_size (same units as delta(s))
    :param units: The units to plot the graph in. Choose one of: pkpc, cMpch, cMpc or pMpc
    :param N: The number of cells in the array that produced the correlation data (used for unit conversions)
    :param log_scale_x: Plot the x-axis as a log scale?
    :param log_scale_y: Plot the y-axis as a log scale?
    :param x_limit: The max value x can have (minimum is always assumed to be 0)
    :param y_limit: The max value y can have (minimum is always assumed to be -1)
    :param save_name: The name to save the final graph too.
    :param sampling_error: The sampling error on y. """
    cell_to_pMpch = Snapshot.BoxsizecMpch / N
    cell_to_cMpc = Snapshot.BoxsizecMpc / N
    cell_to_pMpc = Snapshot.BoxsizepMpc / N
    cell_to_pkpc = Snapshot.Boxsizepkpc / N

    colours = ['b', 'r', 'g', 'black', 'y', 'm', 'c']
    maxima_x = np.zeros(len(xs))                        # used to find the x and y limits of the plot
    maxima_y = np.zeros(len(ys))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Plot every correlation individually
    for i in xrange(len(xs)):
        print i
        if units == 'cMpch':
            xs[i] *= cell_to_pMpch
            x_units = r"$ \rm cMpch$"
        elif units =='pMpc':
            xs[i] *= cell_to_pMpc
            x_units = r"$ \rm pMpc$"
        elif units =='cMpc':
            xs[i] *= cell_to_cMpc
            x_units = 'cMpc'
        elif units == 'pkpc':
            xs[i] *= cell_to_pkpc
            x_units = r"$ \rm pkpc$"
        else:
            x_units = 'cell number'

        # Bin data
        if bin:
            #x, y, x_err, y_err = bin_data_x_space(xs[i], ys[i], bin_size, np.sqrt(2)/2)
            x, y, x_err, y_err = bin_data_quick(np.copy(xs[i]), np.copy(ys[i]), bin_size, np.sqrt(2)/2)
        else:
            x = xs[i]
            y = ys[i]
            x_err = np.sqrt(2)/2
            y_err = 0

        # Add sampling error to y_error
        y_err += sampling_error

        # Set labels and log scale
        ax.set_xlabel(r"$ \rm |\Delta| $"+"$ \ ($"+x_units+"$)$", labelpad=3)
        ax.set_ylabel(r"$ \rm \xi (\Delta) $", labelpad=2)
        if log_scale_x:
            ax.semilogx(nonposx='clip')
            ax.set_xlabel(r"$ \rm  \ |\Delta| $"+"$ \ ($"+x_units+"$)$", labelpad=3)
        if log_scale_y:
            ax.set_ylabel(r"$ \rm \ \xi (\Delta) $", labelpad=2)
            ax.semilogy(nonposy='clip')

        ax.xaxis.set_major_formatter(ScalarFormatter())             # Set scalar numbers on tick labels, not 10^X
        ax.yaxis.set_major_formatter(ScalarFormatter())

        ax.minorticks_on()                                          # Increase size of ticks
        ax.tick_params('both', length=6, width=1, which='major')
        ax.tick_params('both', length=3, width=1, which='minor')

        colour = colours[i % len(colours)]

        # Plot the data
        ax.plot(x, y, color=colour, linewidth=2, marker='.', label=labels[i])

        # Plot the lower and upper limits
        ax.fill_between (x, y-y_err, y+y_err, alpha=0.2, antialiased=True, linestyle='dashed', facecolor=colour, edgecolor=colour)
        ax.fill_betweenx(y, x-x_err, x+x_err, alpha=0.2, antialiased=True, facecolor=colour, edgecolor=colour)

        maxima_x[i] = max(x)
        maxima_y[i] = max(y)+max(y_err)

    # Set axis limits
    if x_limit is not None:
        ax.set_xlim((x_limit[0], x_limit[1]))
    else:
        ax.set_xlim((min(x), 600))
    if y_limit is not None:
        ax.set_ylim((y_limit[0], y_limit[1]))
    else:
        ax.set_ylim(1, max(maxima_y))

    # Plot a horizontal line at y=0 and others if pr
    plt.axhline(y=0, color='black', linestyle='--')

    if horizontal_line is not None:
        plt.axhline(y=horizontal_line, color='r')

    if legend == True:
        plt.legend(frameon=False)

    plt.savefig(save_name+".pdf", dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.savefig(save_name+".png", dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.savefig(save_name+".jpg", dpi=300, bbox_inches='tight', pad_inches=0.04)

    plt.show()

# RUN FUNCTIONS ****************************************************************************************************** #
if __name__ == '__main__':
    grid = np.load(grid_name)
    grid = mask_all_but_LLS(grid)
    x, y = auto_correlation_two_point_fourier(grid)
    plot_correlation(x, y, log_scale_y=True, N=grid.shape[0], sampling_error=0)
