from numpy import arange, zeros, concatenate, sqrt, array, where, repeat, pi
from ..mcmc import mcmc2d
from .model import model

def model_fit(data, funct=['point'], nsteps=1e3, niter=3):

    funct = array(funct)

    flux0 = data.amp[data.uvdist < 30].mean()/funct.size
    fluxstd = data.amp[data.uvdist < 30].std()/funct.size

    # First do a coarse grid search to find the location of the minimum.

    print("Doing coarse grid search.")

    x = arange(-5,5,0.1)
    y = arange(-5,5,0.1)

    params = array([])

    for k in range(funct.size):
        chisq = zeros((x.size,y.size))

        def calc_chisq(d, params, funct):
            model = model(d.u, d.v, params, return_type='data', \
                    funct=funct)

            return ((d.real-model.real)**2*d.weights+(d.imag-model.imag)**2* \
                    d.weights).sum()

        for i in range(x.size):
            for j in range(y.size):
                if (funct[k] == 'point'):
                    par = concatenate((params,array([x[i],y[j],flux0])))
                elif (funct[k] == 'gauss'):
                    par = concatenate((params,array([x[i],y[j],0.1,0.1,0.0, \
                            flux0])))

                chisq[i,j] = calc_chisq(data,par,funct[0:k+1])
    
        xmin = x[where(chisq == chisq.min())[0][0]]
        ymin = y[where(chisq == chisq.min())[1][0]]

        if (funct[k] == 'point'):
            params = concatenate((params,array([xmin,ymin,flux0])))
        elif (funct[k] == 'gauss'):
            params = concatenate((params,array([xmin,ymin,0.1,0.1,0.0,flux0])))

    # Next do a few iterations of MCMC to get the correct solution.

    x = data.u
    y = data.v
    z = concatenate((data.real,data.imag))[:,0]
    sigma_z = 1./sqrt(concatenate((data.weights,data.weights)))[:,0]

    args = {'return_type':'append', 'funct':funct}

    sigma = array([])
    for k in range(funct.size):
        if (funct[k] == 'point'):
            sigma = concatenate((sigma,array([0.1,0.1,fluxstd])))
        elif (funct[k] == 'gauss'):
            sigma = concatenate((sigma,array([0.1,0.1,0.05,0.05,2*pi/10, \
                    fluxstd])))

    temp = {"limited":[False,False], "limits":[0.0,0.0]}
    limits = []
    for i in range(params.size):
        limits.append(temp.copy())
        if ((i == 4) or (i == 10)):
            limits[i]["limited"] = [True, True]
            limits[i]["limits"] = [0.0, pi/2]
        elif ((i == 2) or (i == 3) or (i == 8) or (i == 9)):
            limits[i]["limited"] = [True, False]
            limits[i]["limits"] = [0.0, 0.0]
    limits = array(limits)

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
