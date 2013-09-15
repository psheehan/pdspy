from .calc_opac import calc_opac
from numpy import arange,log10,zeros
from scipy.integrate import trapz

def adist_opac(p,amin,amax,grain,coat=None):
    
    na = 100
    agrid = 10**((log10(amax)-log10(amin))/(na-1)*arange(na)+log10(amin))
    kabsgrid=zeros((grain.lam.size,na))
    kscagrid=zeros((grain.lam.size,na))
    
    normfunc = agrid**(3-p)
    
    for i in range(na):
        grain.a = agrid[i]
        if coat != None:
            coat.a = (1.5*agrid[i]**3)**(1.0/3)
        
        kabs,ksca = calc_opac(grain,coat=coat)
        
        kabsgrid[:,i] = kabs*normfunc[i]
        kscagrid[:,i] = ksca*normfunc[i]
    
    if coat != None:
        agrid *= 1.5**(1.0/3)
    
    norm = trapz(normfunc,x=agrid)
    
    kabs = trapz(kabsgrid,x=agrid)/norm
    ksca = trapz(kscagrid,x=agrid)/norm
    
    return kabs,ksca
