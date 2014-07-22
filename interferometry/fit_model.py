import numpy
from .model import model
from ..mcmc import mcmc2d

def fit_model(data, funct='point', nsteps=1e3, niter=3):

    if type(funct) == str:
        funct = numpy.array([funct])
    elif type(funct) == list:
        funct = numpy.array(funct)
    elif type(funct) == numpy.ndarray:
        pass

    flux0 = data.amp[data.uvdist < 30000.].mean() / funct.size
    fluxstd = data.amp[data.uvdist < 30000.].std() / funct.size

    # First do a coarse grid search to find the location of the minimum.

    print("Doing coarse grid search.")

    x = numpy.arange(-5,5,0.1)
    y = numpy.arange(-5,5,0.1)

    params = numpy.array([])

    for k in range(funct.size):
        chisq = numpy.zeros((x.size, y.size))

        def calc_chisq(d, params, funct):
            m = model(d.u, d.v, params, return_type='data', funct=funct)

            return ((d.real - m.real)**2 * d.weights + \
                    (d.imag - m.imag)**2 * d.weights).sum()

        for i in range(x.size):
            for j in range(y.size):
                if (funct[k] == 'point'):
                    par = numpy.concatenate((params, numpy.array([x[i], \
                            y[j], flux0])))
                elif (funct[k] == 'gauss'):
                    par = numpy.concatenate((params, numpy.array([x[i], \
                            y[j], 0.1, 0.1, 0.0, flux0])))

                chisq[i,j] = calc_chisq(data, par, funct[0:k+1])
    
        xmin = x[numpy.where(chisq == chisq.min())[0][0]]
        ymin = y[numpy.where(chisq == chisq.min())[1][0]]

        if (funct[k] == 'point'):
            params = numpy.concatenate((params, numpy.array([xmin, \
                    ymin, flux0])))
        elif (funct[k] == 'gauss'):
            params = numpy.concatenate((params, numpy.array([xmin, ymin, \
                    0.1, 0.1, 0.0, flux0])))

    # Next do a few iterations of MCMC to get the correct solution.

    x = data.u
    y = data.v
    z = numpy.concatenate((data.real, data.imag))[:,0]
    sigma_z = 1./numpy.sqrt(numpy.concatenate((data.weights,data.weights)))[:,0]

    args = {'return_type':'append', 'funct':funct}

    sigma = numpy.array([])
    for k in range(funct.size):
        if (funct[k] == 'point'):
            sigma = numpy.concatenate((sigma, numpy.array([0.1, 0.1, fluxstd])))
        elif (funct[k] == 'gauss'):
            sigma = numpy.concatenate((sigma, numpy.array([0.1, 0.1, 0.05, \
                    0.05, 2*numpy.pi/10, fluxstd])))

    temp = {"limited":[False,False], "limits":[0.0,0.0]}
    limits = []
    for i in range(funct.size):
        if funct[i] == 'point':
            limits.append(temp.copy())
            limits.append(temp.copy())
            limits.append({"limited":[True,False], "limits":[0.0,0.0]})
        elif funct[i] == 'gauss':
            limits.append(temp.copy())
            limits.append(temp.copy())
            limits.append({"limited":[True,False], "limits":[0.0,0.0]})
            limits.append({"limited":[True,False], "limits":[0.0,0.0]})
            limits.append({"limited":[True,True], "limits":[0.0,numpy.pi/2]})
            limits.append({"limited":[True,False], "limits":[0.0,0.0]})
    limits = numpy.array(limits)

    for i in range(niter):
        if (i == niter-1):
            nsteps *= 100

        print("Doing MCMC iteration #", i+1, "with", nsteps, "steps.")

        accepted_params = mcmc2d(x, y, z, sigma_z, params, sigma, \
                model, args=args, nsteps=nsteps, limits=limits)

        accepted_params = accepted_params[int(accepted_params.shape[0]*0.25):,:]

        params = accepted_params.mean(axis=0)
        sigma = accepted_params.std(axis=0)

    return params, sigma, accepted_params
