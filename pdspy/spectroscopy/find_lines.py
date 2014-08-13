from numpy import arange,array,where
from numpy import abs as absv
from .line_flux import line_flux
import matplotlib.pyplot as plt

def find_lines(data,thresh=3):
    
    nside = 8
    nleft = 8
    nright = 8
    
    index = 8
    
    B = 1.0/1200
    
    while index < data.wave.size-8:
        
        lines = array([data.wave[index]])
        
        results,chisq = line_flux(data,lines,nleft=nleft,nright=nright, \
            quiet=True)
        results_fw,chisq_fw = line_flux(data,lines,nleft=nleft,nright=nright,\
            quiet=True,fixed_width=True)
        
        chisq_test = (chisq <= thresh) & (chisq_fw <= thresh)
        SN_test = (results[0,1]/results[0,2] >= 5) & \
                  (results_fw[0,1]/results_fw[0,2] >= 5)
        width_test = (results[0,3] >= lines[0]*B/2) & (results[0,3] <= \
            lines[0]*3*B)
        flux_test = absv(results[0,1]-results_fw[0,1])/results[0,1] <= 0.2
        
        #if (chisq_test & flux_test) & (SN_test & width_test):
        if (chisq_test & SN_test):
            print("    {0:f}   {1:e}   {2:e}   {3:f}   {4:f}".format( \
                results[0,0],results[0,1],results[0,2],results[0,3],chisq))
            
            results,chisq = line_flux(data,lines,nleft=nleft,nright=nright, \
                plotout="{0:6.3f}.pdf".format(results[0,0]),quiet=True)
            
            index = where(absv(data.wave-data.wave[index]-0.05) == \
                absv(data.wave-data.wave[index]-0.05).min())[0][0]
            nside = 6
        else:
            index = where(absv(data.wave-data.wave[index]-0.05) == \
                absv(data.wave-data.wave[index]-0.05).min())[0][0]
#        else:
#            if (nleft > 3) & (nright == nside):
#                nleft = nleft - 1
#            else:
#                nleft = nside
#                if (nright > 3) & (nleft == nside):
#                    nright = nright-1
#                else:
#                    nright = nside
#                    if nside > 3:
#                        nside = nside - 1
#                        nleft = nside
#                        nright = nside
#                    else:
#                        nside = 8
#                        nleft = nside
#                        nright = nside
#                        index = where(absv(data.wave-data.wave[index]-0.05) ==\
#                            absv(data.wave-data.wave[index]-0.05).min())[0][0]
