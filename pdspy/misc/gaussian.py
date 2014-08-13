from numpy import exp

def gaussian(x,x0,sigma,f0):
    
    return f0*exp(-1*(x-x0)**2/(2*sigma**2))