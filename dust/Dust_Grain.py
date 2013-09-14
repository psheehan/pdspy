class Dust_Grain:
    
    def __init__(self,a,rho,lam=None,n=None,k=None,kabs=None,ksca=None):
        self.a = a
        self.rho = rho
        self.lam = lam
        self.n = n
        self.k = k
        if (n != None) & (k != None):
            self.m = n+1j*k
        self.kabs = None
        self.ksca = None
