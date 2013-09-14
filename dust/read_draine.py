from numpy import loadtxt,flipud

def read_draine(filename):
    
    opt_data = loadtxt(filename)
    
    lam = flipud(opt_data[:,0])*1.0e-4 # in cm
    n = flipud(opt_data[:,3])+1.0
    k = flipud(opt_data[:,4])
    
    return lam,n,k
