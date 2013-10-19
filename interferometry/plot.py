import matplotlib.pyplot as plt
from numpy import array,sqrt

def plot(data,xaxis,yaxis,filename=None,channel=None,ploterr=False, \
    xrange=None,yrange=None):
    
    if channel == None:
        channel=array([1,data.real.shape[1]])

    real = data.real[:,channel[0]-1:channel[1]]
    imag = data.imag[:,channel[0]-1:channel[1]]
    weights = data.weights[:,channel[0]-1:channel[1]]
    amp = data.amp[:,channel[0]-1:channel[1]]
    phase = data.phase[:,channel[0]-1:channel[1]]
    u = data.u
    v = data.v
    uvdist = data.uvdist

    weights[weights <= 0] = 0
    
    if xaxis == "u":
        x = u/1000.
        xtitle = "U (k$\lambda$)"
    elif xaxis == "v":
        x = v/1000
        xtitle = "V (k$\lambda$)"
    elif xaxis == "uvdist":
        x = uvdist/1000
        xtitle = "Baseline (k$\lambda$)"
    
    if yaxis == "u":
        y = u/1000
        ytitle = "U (k$\lambda$)"
    elif yaxis == "v":
        y = v/1000
        ytitle = "V (k$\lambda$)"
    elif yaxis == "uvdist":
        y = uvdist/1000
        ytitle = "Baseline (k$\lambda$)"
    elif yaxis == "real":
        y = real
        ytitle = "Real (Jy)"
    elif yaxis == "imag":
        y = imag
        ytitle = "Imaginary (Jy)"
    elif yaxis == "amp":
        y = amp
        ytitle = "Amplitude (Jy)"
    elif yaxis == "phase":
        y = phase
        ytitle = "Phase (rad)"
    
    if xrange == None:
        if xaxis == "uvdist": 
            xmin=0.0
        else:
            xmin = x.min()*1.05
        xmax = x.max()*1.05
        xrange = array([xmin,xmax])
    
    if yrange == None:
        if yaxis == "amp":
            ymin = 0
        else:
            ymin = y.min()*1.05
        ymax = y.max()*1.05
        yrange = array([ymin,ymax])
    
    if ploterr == False:
        if (yaxis == "u") ^ (yaxis == "v") ^ (yaxis == "uvdist"):
            plt.plot(x,y,'o')
        else:
            for i in range(channel[1]-channel[0]+1):
                good = weights[:,i] >= 0.0
                plt.plot(x[good],y[good,i],'o')
    else:
        for i in range(channel[1]-channel[0]+1):
            good = weights[:,i] >= 0.0
            plt.errorbar(x[good],y[good,i],yerr=1/sqrt(weights[good,i]),fmt='o')
    
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.axis([xrange[0],xrange[1],yrange[0],yrange[1]])
    
    if filename != None:
        plt.savefig(filename)
    else:
        plt.show()
    
    plt.close()
