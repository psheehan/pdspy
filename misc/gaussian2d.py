from numpy import exp,cos,sin

def gaussian2d(x,y,x0,y0,sigmax,sigmay,pa,f0):
    
    xp=(x-x0)*cos(pa)-(y-y0)*sin(pa)
    yp=(x-x0)*sin(pa)+(y-y0)*cos(pa)
    
    return f0*exp(-1*xp**2/(2*sigmax**2))*exp(-1*yp**2/(2*sigmay**2))