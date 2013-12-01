from .average import average
from .model import model
from .interferometry import Visibilities
from numpy import array

def center(data,params):
    
    params=array(params)
    
    if params.size > 2:
        params[2] = 1.0
    
    mod = model(data.u,data.v,params,funct=array(["point"]), \
                    return_type="data")
    
    data_complex = data.real+1j*data.imag
    model_complex = mod.real+1j*mod.imag
    
    centered_data = data_complex * model_complex.conj()
    
    return Visibilities(data.u,data.v,data.freq,centered_data.real, \
        centered_data.imag,data.weights)
