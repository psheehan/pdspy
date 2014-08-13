from numpy import zeros
from numpy.random import randint, normal

def change_params(params, sigma_params, change_param=None):

    new_params = zeros(params.size)

    if change_param == None:
        change_param = randint(low=0, high=params.size, size=1)[0]

    new_params = params.copy()
    new_params[change_param] = params[change_param] + \
        normal(loc=0., scale=sigma_params[change_param], size=1)[0]

    return new_params
