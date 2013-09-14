import matplotlib.pyplot as plt

def plot_spectrum(spectrum,dpc=1,xlog=False, ylog=False):
    
    plt.plot(spectrum.wave,spectrum.flux*(1.0/dpc)**2)
    
    if xlog:
        plt.gca().set_xscale("log")
    if ylog:
        plt.gca().set_yscale("log")
    
    plt.xlabel("$\lambda$ [$\mu$m]")
    plt.ylabel("F$_{\mu}$ [Jy]")
    
    plt.show()