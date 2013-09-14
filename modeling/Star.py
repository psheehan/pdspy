import numpy
from ..constants.physics import sigma

class Star:
    def __init__(self, mstar, lstar, tstar, xstar=0.0, ystar=0.0, zstar=0.0):
        self.mstar = mstar
        self.lstar = lstar
        self.tstar = tstar
        self.rstar = (lstar/(4*numpy.pi*sigma*tstar**4))**(1./2)
