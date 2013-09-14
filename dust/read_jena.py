from numpy import loadtxt,flipud

def read_jena(filename,type="standard"):
    
    opt_data = loadtxt(filename)
    
    if type == "standard":
        lam = flipud(1./opt_data[:,0]) # in cm
        n = flipud(opt_data[:,1])
        k = flipud(opt_data[:,2])
    elif type == "umwave":
        lam = flipud(opt_data[:,0])*1.0e-4 # in cm
        n = flipud(opt_data[:,1])
        k = flipud(opt_data[:,2])
    
    return lam,n,k