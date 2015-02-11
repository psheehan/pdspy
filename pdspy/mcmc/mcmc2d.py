from numpy import array, arange, exp
from numpy.random import uniform
from .change_params import change_params
from .ml2d import ml2d

def mcmc2d(x, y, z, sigma_z, params, sigma_params, model, args={}, \
           nsteps=1e5, change_param=None, limits=None):

    MLold, chisq_old = ml2d(x, y, z, sigma_z, params, model, args=args, \
            limits=limits)

    accepted_params = []

    for i in arange(nsteps):
        new_params = change_params(params, sigma_params, \
                change_param=change_param)

        MLnew, chisq_new = ml2d(x, y, z, sigma_z, new_params, model, \
                args=args, limits=limits)

        if chisq_new < chisq_old:
            accepted_params.append(params)
            params = new_params
            MLold = MLnew
            chisq_old = chisq_new
        else:
            if uniform(low=0., high=1., size=1)[0] <= \
                    exp(0.5*(chisq_old-chisq_new)):
                accepted_params.append(params)
                params = new_params
                MLold = MLnew
                chisq_old = chisq_new

    return array(accepted_params)
