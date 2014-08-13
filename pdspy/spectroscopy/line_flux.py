from ..constants.physics import c
from ..constants.math import pi
from ..constants.astronomy import Jy
from numpy import arange,ones,concatenate,sqrt,exp,where,array,mat,sin,log
from numpy import abs as absv
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

def line_flux(data,lines,nleft=8,nright=8,plotout=None,quiet=False, \
    fixed_width=False,fringing=False):
    
    B = 1.0/1500
    
    lines = array(lines)
    
    wave = data.wave
    flux = data.flux
    unc = data.unc
    
    nlines = lines.size
    ind = arange(nlines)
    
    left = where(absv(wave-lines[0]) == absv(wave-lines[0]).min())
    right = where(absv(wave-lines[lines.size-1]) == absv(wave- \
        lines[lines.size-1]).min())
    
    prange = arange(right[0]-left[0]+nleft+nright+1)-nleft+left[0]
    
    wave_fit = wave[prange]
    flux_fit = flux[prange]
    unc_fit = unc[prange]
    
    flux_in = ones(lines.size)
    #sigma_in = lines*B
    fwhm_in = lines/(888.488-9.553*lines)
    sigma_in = fwhm_in/(2*sqrt(2*log(2)))
    
    omega_in = array([2*pi/(4*sigma_in[0])])
    amp_in = array([0.1])
    phi_in = array([pi/2])
    
    slope = array([(flux_fit[flux_fit.size-1]-flux_fit[0])/(wave_fit.max()- \
        wave_fit.min())])
    yint = array([flux_fit[0]-slope[0]*wave_fit.min()])
    
    if fringing:
        A = concatenate((flux_in,lines,sigma_in,yint,slope, \
                amp_in,omega_in,phi_in))
    else:
        A = concatenate((flux_in,lines,sigma_in,yint,slope))
    
    parinfo = []
    for i in arange(A.size):
        parinfo.append({"limited":[False,False], "limits":[0.0,0.0], \
            "mpside":2, "fixed":False})    
    
    for i in arange(lines.size):
        parinfo[i]["limited"] = [True,False]
        parinfo[i]["limits"] = [0.001,0.0]
        parinfo[lines.size+i]["limited"] = [True,True]
        parinfo[lines.size+i]["limits"] = [lines[i]-0.025,lines[i]+0.025]
        parinfo[2*lines.size+i]["limited"] = [True,False]
        parinfo[2*lines.size+i]["limits"] = [1.0e-3,0.0]
        if fixed_width:
            parinfo[2*lines.size+i]["fixed"] = True
    
    if fringing:
        parinfo[-1]["limited"] = [True,True]
        parinfo[-1]["limits"] = [0.0,2*pi]
    
    fa = {"x":wave_fit, "y":flux_fit, "err":unc_fit, "fringing":fringing}
    mfit = mpfit(gauss,xall=A,functkw=fa,parinfo=parinfo,quiet=1)
    
    A=mfit.params
    
    fit = gauss(A,x=wave_fit,y=flux_fit,err=unc_fit,fringing=fringing)[1]* \
            unc_fit*(-1)+flux_fit
    chisq = ((fit - flux_fit)**2/unc_fit**2).sum()/(wave_fit.size-A.size)
    
    # Calculate the flux and uncertainty.
    
    F = sqrt(2*pi)*A[ind]*Jy*(c*A[ind+2*nlines]*1.0e-4)/(A[ind+nlines]* \
        1.0e-4)**2/1.0e7
    
    deltaF = ones(nlines)
    for i in arange(nlines):
        deltaF[i] = sqrt(((unc_fit*Jy*c*B/(wave_fit*1.0e-4)*exp(-1.0*( \
            wave_fit-A[i+nlines])**2/(2*A[i+2*nlines]**2)))**2).sum())/1.0e7
    
    # Output the results.
    
    Results = concatenate((array(mat(A[ind+nlines]).T),array(mat(F).T), \
        array(mat(deltaF).T),array(mat(A[ind+2*nlines]*2*sqrt(2*log(2))).T)), \
        axis=1)
    
    if quiet == False:
        print("")
        for i in arange(lines.size):
            print("  {0:>6.3f}  {1:>9.3e}  {2:>9.3e}  {3:>6.4f}".format( \
                Results[i,0],Results[i,1],Results[i,2],Results[i,3]))
        print("")
        print("Reduced chi-squared of the fit: ",chisq)
        print("")
    
    # Plot the results.
    
    if (plotout != None) or (quiet == False):
        plt.errorbar(wave_fit,flux_fit,fmt="b",yerr=unc_fit)
        plt.plot(wave_fit,fit,"r")
        plt.xlabel("$\lambda$ [$\mu$"+"m]")
        plt.ylabel(r"F$_{\nu}$ [Jy]")
        if plotout != None:
            plt.savefig(plotout)
        elif quiet == False:
            plt.show()
    
    plt.clf()
    
    return Results, chisq

def gauss(p, fjac=None, x=None, y=None, err=None, fringing=False):
    
    if fringing:
        model = p[p.size-5]+p[p.size-4]*x+ \
                p[p.size-3]*sin(p[p.size-2]*x+p[p.size-1])
    
        for i in arange(p.size/3-1):
            model += p[i]*exp(-1*(x-p[p.size/3-1+i])**2/ \
                        (2*p[2*(p.size/3-1)+i]**2))
    else:
        model = p[p.size-2]+p[p.size-1]*x

        for i in arange(p.size/3):
            model += p[i]*exp(-1*(x-p[p.size/3+i])**2/ \
                        (2*p[2*(p.size/3)+i]**2))
    
    status = 0
    
    return [status, (y-model)/err]
