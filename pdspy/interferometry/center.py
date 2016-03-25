from .model import model
from .libinterferometry import Visibilities
import numpy

def center(data, params):
    
    if type(params) == list:
        params = numpy.array(params)
    elif type(params) == numpy.ndarray:
        pass
    
    if params.size > 2:
        params[2] = 1.0
    
    data_complex = data.real+1j*data.imag
    model_complex = model(data.u, data.v, params, funct="point", \
            return_type="complex")
    
    centered_data = data_complex * model_complex.conj()
    
    return Visibilities(data.u.copy(), data.v.copy(), data.freq.copy(), \
            centered_data.real, centered_data.imag, data.weights.copy())
