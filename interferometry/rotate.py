from .interferometry import Visibilities
from numpy import cos, sin

def rotate(data,pa=0):

    newu =  data.u*cos(pa) + data.v*sin(pa)
    newv = -data.u*sin(pa) + data.v*cos(pa)

    return Visibilities(newu,newv,data.freq,data.real,data.imag,data.weights)
