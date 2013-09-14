from .density import protoplanetary_disk

class Disk:
    def __init__(self, mass=1.0e-3, rmin=0.1, rmax=300, plrho=2.37, h0=0.1, \
            plh=58./45., dust="dustkappa_yso.inp"):
        self.mass = mass
        self.rmin = rmin
        self.rmax = rmax
        self.plrho = plrho
        self.h0 = h0
        self.plh = plh
        self.dust = dust

    def density(self, r, theta):
        return protoplanetary_disk(r, theta, mass=self.mass, rin=self.rmin, \
                rout=self.rmax, plrho=self.plrho, h=self.h0, plh=self.plh)
