class NumberUnit:

    def __init__(self,cgs):
        self.cgs = cgs

    def __add__(self,a):
        return self.cgs + a

    def __radd__(self,a):
        return self.cgs + a

    def __sub__(self,a):
        return self.cgs - a

    def __rsub__(self,a):
        return a - self.cgs

    def __mul__(self,a):
        return self.cgs * a

    def __rmul__(self,a):
        return self.cgs * a

    def __div__(self,a):
        return self.cgs / a

    def __rdiv__(self,a):
        return a / self.cgs
