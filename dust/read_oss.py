from numpy import loadtxt

def read_oss(filename):
    
    opt_data = loadtxt(filename)
    
    lam = opt_data[:,0] # in cm
    n = opt_data[:,1]
    k = opt_data[:,2]
    
    return lam,n,k
