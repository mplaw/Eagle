""" Hard-codes default snapshot-parameters, but also has the power to import different values.
    Creates 'Snapshot' object from which everything can be accessed, e.g. Snapshot.BoxsizecMpch
    Code: Python 2.7.9
    Requirements: h5py, numpy
    Parameters:
        BoxsizecMpch    : The size of the simulation box            :: cMpc/h
        BoxsizecMpc     : The size of the simulation box            :: cMpc
        BoxsizepMpc     : The size of the simulation box            :: pMpc
        Boxsizepkpc     : The size of the simulation box            :: pkpc
        Ez              : Unknown                                   :: unknown
        ExpansionFactor : More commonly denoted by a                :: unitless
        Hz              : Hubble paramter at redshift = Redshift    :: 1/s
        H0              : Hubble paramter at redshift = 0           :: cms^-1/Mpc
        HubbleParam     : H0 / 100                                  :: cms^-1/Mpc
        Omega0          : Normal + dark matter fraction             :: unitless
        OmegaBaryon     : Baryon mass fraction                      :: unitless
        OmegaLambda     : Dark energy fraction                      :: unitless
        Redshift        : Obvious                                   :: unitless
        U_L             : Unit of length in simulation              :: cm
        U_M             : Unit of Mass in simulation                :: g
        p_c             : Critical density                          :: g/cm^3

    Functions:
        calculateParameters(self)   : Calculates p_c, and box sizes
        importSnap(self, file)      : Takes the file name of a HDF5 snapshot, file, and imports it's parameters
        si                          : convert units to si
        cgs                         : Convert units to cgs
"""
from __future__ import division
import numpy as np
import h5py
import os


__author__ = 'Matt'


pi = np.pi


def importAttrs(filePath, hdfPath = 'Header'):
    """ Import an attribute from a HDF5 file. """
    f               = h5py.File(filePath)
    folder          = f[hdfPath]
    values          = folder.attrs.values()
    keys            = folder.attrs.keys()
    f.close()
    return values, keys


class SNAPSHOT:
    """ Hold all snapshot-related information. Use default values for snap_012_z003p017 unless import is called, in which case import
    a new snapshots' information. All default units are cgs. """
    BoxsizecMpch    = 8.47125               # Comoving size of box      :: cMpc / h
    BoxsizecMpc     = None                  # "------------------"      :: cMpc
    BoxsizepMpc     = None                  # Physical size of box      :: pMpc
    Boxsizepkpc     = None                  # "------------------"      :: pkpc
    Boxsizeckpc     = None                  # Comoving size of box      :: ckpc
    Ez              = 4.537093477630583     # ...
    ExpansionFactor = 0.2489726990748467    # a
    Hz              = 9.964708881888158E-18 # Hubble parameter z=3      :: s^-1
    H0              = 6777000               # Hubble parameter z=0      :: cms^-1/Mpc
    HubbleParam     = 0.6777                # H_0/100                   ::
    Omega0          = 0.307                 # normal + dark matter mass fraction
    OmegaBaryon     = 0.0482519             # Baryon
    OmegaLambda     = 0.693                 # Dark energy fraction
    Redshift        = 3.0165046357126
    U_L             = 3.085678E24           # Unit Length - 1 Mpc       :: cm
    U_M             = 1.989E43              # Unit Mass - 10^10*M_sun   :: g
    p_c             = None
    units           = 'cgs'                 # Unit type: 'cgs' or 'si'
    folder_name     = 'D:/L4Law/L4Law/Eagle/Snap'
    urchin_name     = 'D:/L4Law/L4Law/Eagle/Urchin'

    def calculateParameters(self):
        self.BoxsizecMpc     = self.BoxsizecMpch/self.HubbleParam       # "------------------"      :: cMpc
        self.BoxsizepMpc     = self.BoxsizecMpc*self.ExpansionFactor    # Physical size of box      :: pMpc
        self.Boxsizepkpc     = self.BoxsizepMpc*1000                    # "------------------"      :: pkpc
        self.Boxsizeckpc     = self.BoxsizecMpc*1000                    # Comoving size of box      :: ckpc
        self.p_c             = 3*(self.Hz**2)/(8*pi*6.67384e-8)

    def __init__(self):
        self.calculateParameters()

    def importSnap(self, file, urchin_file):
        self.folder_name = file
        self.urchin_name = urchin_file

        file_names = os.listdir(file)
        attVal, attKeys = importAttrs(file + "/" + file_names[0], 'Header')
        Q               = zip(attKeys, attVal)
        P               = dict(Q)
        self.BoxsizecMpch    = P['BoxSize']
        self.ExpansionFactor = P['ExpansionFactor']
        self.Ez              = P['E(z)']
        self.Hz              = P['H(z)']
        self.HubbleParam     = P['HubbleParam']
        self.Omega0          = P['Omega0']
        self.OmegaBaryon     = P['OmegaBaryon']
        self.OmegaLambda     = P['OmegaLambda']
        self.Redshift        = P['Redshift']
        self.calculateParameters()
        del attVal, attKeys, Q, P

    def setFolder(self, folder):
        self.folder_name = folder

    def si(self):
        if self.units != 'si':
            self.U_L             = 3.085678E22
            self.U_M             = 1.989E40
            self.H0              = 67.77
            self.p_c             = 3*(self.Hz**2)/(8*pi*6.67384e-11)

    def cgs(self):
        if self.units != 'cgs':
            self.U_L             = 3.085678E24
            self.U_M             = 1.989E43
            self.H0              = 6777000
            self.p_c             = 3*(self.Hz**2)/(8*pi*6.67384e-8)


Snapshot = SNAPSHOT()
