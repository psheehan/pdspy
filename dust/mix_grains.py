from numpy import array,zeros,sqrt
from .Dust_Grain import Dust_Grain
from scipy.optimize import fsolve

def mix_grains(a,grains,abund,medium=None,rule="Bruggeman"):
    
    pi = 3.1415927
    
    if rule == "Bruggeman":
        meff = zeros(grains[0].lam.size,dtype=complex)
        rho = 0.0
        
        for i in range(grains[0].lam.size):
            temp = fsolve(bruggeman,array([1.0,0.0]),args=(grains,abund,i))
            meff[i] = temp[0]+1j*temp[1]
        
        for i in range(grains.size):
            rho += grains[i].rho*abund[i]
    
    elif rule == "MaxGarn":
        sigma = 0.0+1j*0.0
        rho = 0.0
        
        for i in range(grains.size):
            sigma += abund[i]*(grains[i].m**2-medium.m**2)/(grains[i].m**2+2*medium.m**2)
            rho += grains[i].rho*abund[i]
        
        meff = sqrt(medium.m**2*(1+3*sigma/(1-sigma)))
    
    return Dust_Grain(a,rho,lam=grains[0].lam,n=meff.real,k=meff.imag)

def bruggeman(meff,grains,abund,index):
    
    m_eff = meff[0]+1j*meff[1]
    tot = 0+0j
    
    for j in range(grains.size):
        tot += abund[j]*(grains[j].m[index]**2-m_eff**2)/(grains[j].m[index]**2+2*m_eff**2)
    
    return array([tot.real,tot.imag])
