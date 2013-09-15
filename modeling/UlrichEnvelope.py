from .density import ulrich_envelope

class UlrichEnvelope:
    def __init__(self, mass=1.0e-3, rmin=0.1, rmax=1000, rcent=30, cavpl=1.0, \
            cavrfact=0.2, dust="dustkappa_yso.inp"):
        self.mass = mass
        self.rmin = rmin
        self.rmax= rmax
        self.rcent = rcent
        self.cavpl = cavpl
        self.cavrfact = cavrfact
        self.dust = dust

    def density(self, r, theta):
        return ulrich_envelope(r, theta, rin=self.rmin, rout=self.rmax, \
                mass=self.mass, rcent=self.rcent, cavpl=self.cavpl, \
                cavrfact=self.cavrfact)
