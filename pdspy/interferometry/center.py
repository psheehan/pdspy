from .model import model
from .libinterferometry import Visibilities
import numpy

def center(data, params):
    
    if type(params) == list:
        params = numpy.array(params)
    elif type(params) == numpy.ndarray:
        pass
    
    centering_params = [params[0], params[1], 1.]
    
    data_complex = data.real+1j*data.imag

    model_complex = numpy.empty(data.real.shape, dtype=complex)
    for i in range(len(data.freq)):
        model_complex[:,i] = model(data.u*data.freq[i]/data.freq.mean(), \
                data.v*data.freq[i]/data.freq.mean(), centering_params, \
                funct="point", return_type="complex")[:,0]
    
    centered_data = data_complex * model_complex.conj()
    
    return Visibilities(data.u.copy(), data.v.copy(), data.freq.copy(), \
            centered_data.real, centered_data.imag, data.weights.copy())
