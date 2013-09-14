from .bhmie import bhmie
from .bhcoat import bhcoat
from numpy import zeros,arange

def calc_opac(grain,coat=None):
    
    pi = 3.1415927
    
    kabs = zeros(grain.lam.size)
    ksca = zeros(grain.lam.size)
    
    if coat == None:
        mdust = 4*pi*grain.a**3/3*grain.rho
        
        for i in arange(grain.lam.size):
            x = 2*pi*grain.a/grain.lam[i]
            
            S1,S2,Qext,Qsca,Qback,gsca=bhmie(x,grain.m[i],1000)
            
            Qabs = Qext - Qsca
            
            kabs[i] = pi*grain.a**2*Qabs/mdust
            ksca[i] = pi*grain.a**2*Qsca/mdust
    
    else:
        mdust = 4*pi*grain.a**3/3*grain.rho+ \
                4*pi/3*(coat.a**3-grain.a**3)*coat.rho
        
        for i in arange(grain.lam.size):
            x = 2*pi*grain.a/grain.lam[i]
            y = 2*pi*coat.a/coat.lam[i]
            
            Qext,Qsca,Qback=bhcoat(x,y,grain.m[i],coat.m[i])
            
            Qabs = Qext - Qsca
            
            kabs[i] = pi*(coat.a)**2*Qabs/mdust
            ksca[i] = pi*(coat.a)**2*Qsca/mdust
    
    return kabs,ksca
