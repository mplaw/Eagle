""" Contains all physical constants in an object called 'Constant', with some methods to change their units. Imports directly into global namespace.
    Constants (default units are cgs):
        pi
        exp     : Euler's number
        c       : Speed of light
        e       : Elementary charge
        k_B     : Boltzmann constant
        M0      : Mass of sun
        L_sun   : Luminosity of sun
        G       : Gravitational constant
        Mpc     : Mega parsecs
        planck  : Planck's constant
        m_p     : Mass of proton
        m_e     : Mass of electron
        m_HI    : Mass of neutral Hydrogen atom

    Functions
        si : change units to s.i.
        cgs: change units to cgs (centimetres, grams, seconds)
"""
from __future__ import division


# Use object with methods
class Constants:
    """ All physical constants. Default units are cgs, change to s.i. using method si(). """
    pi      = 3.14159265359         # pi                        :: unit less
    exp     = 2.71828182845         # Euler's number            :: unit less
    c       = 2.99792458e10         # Speed of light            :: cm/s
    e       = 4.80320425e-10        # Elementary charge         :: cgs statocoloumbs
    k_B     = 1.3806488e-16         # Boltzmann Constant        :: cm^2 g s^-1 K^-1 (erg K^-1)
    M0      = 1.9891e33             # Sun's mass                :: g
    L_sun   = 3.846e33              # Sun's Luminosity          :: gcm^2s^-3
    G       = 6.67384e-8            # Gravitational constant    :: cm3/g/s2
    Mpc     = 3.08567758e24         # 1 Mpc                     :: cm
    planck  = 6.62606957e-34        # Planck's Constant         :: m^2.kg/s
    m_p     = 1.67262178e-24        # Mass of proton            :: g
    m_e     = 9.10938291e-28        # Mass of electron          :: g
    m_HI    = 1.6737236e-24         # Mass of neutral Hydrogen  :: g
    units   = 'cgs'

    def __init__(self):
        pass

    def si(self):
        if self.units != 'si':
            self.c       = 2.99792458e8
            self.e       = 1.602176565e-19
            self.k_B     = 1.3806488e-23
            self.M0      = 1.9891e30
            self.L_sun   = 3.846e26
            self.G       = 6.67384e-11
            self.Mpc     = 3.08567758e22
            self.planck  = 6.62606957e-34
            self.m_p     = 1.67262178e-27
            self.m_e     = 9.10938291e-31
            self.m_HI    = 1.6737236e-27
            self.units   = 'si'

    def cgs(self):
        if self.units != 'cgs':
            self.c       = 2.99792458e10
            self.e       = 4.80320425e-10
            self.k_B     = 1.3806488e-16
            self.M0      = 1.9891e33
            self.L_sun   = 3.846e33
            self.G       = 6.67384e-8
            self.Mpc     = 3.08567758e24
            self.planck  = 6.62606957e-34
            self.m_p     = 1.67262178e-24
            self.m_e     = 9.10938291e-28
            self.m_HI    = 1.6737236e-24
            self.units   = 'cgs'


Constant = Constants()
