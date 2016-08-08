import numpy
import emcee
import matplotlib.pyplot as plt
from .model import model
from ..mcmc import mcmc2d

def fit_model(data, funct='point', nsteps=1e3, niter=3):

    if type(funct) == str:
        funct = numpy.array([funct])
    elif type(funct) == list:
        funct = numpy.array(funct)
    elif type(funct) == numpy.ndarray:
        pass

    min_baselines = 0.25 * (data.uvdist.max() - data.uvdist.min()) + \
            data.uvdist.min()

    flux0 = data.amp[data.uvdist < min_baselines].mean() / funct.size
    fluxstd = data.amp[data.uvdist < min_baselines].std() / funct.size

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
                elif (funct[k] == 'circle'):
                    par = numpy.concatenate((params, numpy.array([x[i], \
                            y[j], 0.1, 0.0, 0.0, flux0])))
                elif (funct[k] == 'ring'):
                    par = numpy.concatenate((params, numpy.array([x[i], \
                            y[j], 0.1, 0.2, 0.0, 0.0, flux0])))

                chisq[i,j] = calc_chisq(data, par, funct[0:k+1])
    
        xmin = x[numpy.where(chisq == chisq.min())[0][0]]
        ymin = y[numpy.where(chisq == chisq.min())[1][0]]

        if (funct[k] == 'point'):
            params = numpy.concatenate((params, numpy.array([xmin, \
                    ymin, flux0])))
        elif (funct[k] == 'gauss'):
            params = numpy.concatenate((params, numpy.array([xmin, ymin, \
                    0.1, 0.1, 0.0, flux0])))
        elif (funct[k] == 'circle'):
            params = numpy.concatenate((params, numpy.array([xmin, ymin, \
                    0.1, 0.0, 0.0, flux0])))
        elif (funct[k] == 'ring'):
            params = numpy.concatenate((params, numpy.array([xmin, ymin, \
                    0.1, 0.2, 0.0, 0.0, flux0])))

    # Next do a few iterations of MCMC to get the correct solution.

    x = data.u
    y = data.v
    z = numpy.concatenate((data.real, data.imag))[:,0]
    zerr = 1./numpy.sqrt(numpy.concatenate((data.weights,data.weights)))[:,0]

    args = {'return_type':'append', 'funct':funct}

    # Define a likelihood function.

    def lnlike(p, x, y, z, zerr):
        m = model(x, y, params, return_type="append", funct=funct)

        return -0.5*(numpy.sum((z - m)**2 / zerr**2))

    if funct[0] == 'point':
        def lnprior(p):
            if 0. < p[2]:
                return 0.0

            return -numpy.inf
    elif funct[0] == 'gauss':
        def lnprior(p):
            if 0. < p[2] and 0. < p[3] < p[2] and -numpy.pi/2 <= p[4] <= numpy.pi/2 and \
                    0. < p[5]:
                return 0.0

            return -numpy.inf
    elif funct[0] == 'circle':
        def lnprior(p):
            if 0. < p[2] and 0. <= p[3] <= numpy.pi/2 and \
                    0.0 <= p[4] <= numpy.pi and 0. < p[5]:
                return 0.0

            return -numpy.inf
    elif funct[0] == 'ring':
        def lnprior(p):
            if 0. < p[2] and p[2] < p[3] and 0.0 <= p[4] <= numpy.pi/2 and \
                    0 <= p[5] <= numpy.pi and 0. < p[6]:
                return 0.0

            return -numpy.inf

    def lnprob(p, x, y, z, zerr):
        lp = lnprior(p)

        if not numpy.isfinite(lp):
            return -numpy.inf

        return lp + lnlike(p, x, y, z, zerr)

    # Set up the emcee run.

    if funct[0] == 'point':
        ndim, nwalkers = 2, 250

        pos = []
        for i in range(nwalkers):
            pos.append([numpy.random.normal(xmin,0.1,1)[0], \
                    numpy.random.normal(xmin,0.1,1)[0], \
                    numpy.random.uniform(0.0001,1.0,1)[0]])
    elif funct[0] == 'gauss':
        ndim, nwalkers = 6, 250

        pos = []
        for i in range(nwalkers):
            sigma_x = numpy.random.uniform(0.01,1.0,1)[0]

            pos.append([numpy.random.normal(xmin,1.0e-4,1)[0], \
                    numpy.random.normal(xmin,1.0e-4,1)[0], \
                    sigma_x, \
                    numpy.random.uniform(0.01,sigma_x,1)[0], \
                    numpy.random.uniform(-numpy.pi/2, numpy.pi/2, 1)[0], \
                    numpy.random.uniform(0.0001,1.0,1)[0]])
    elif funct[0] == 'circle':
        ndim, nwalkers = 6, 250

        pos = []
        for i in range(nwalkers):
            pos.append([numpy.random.normal(xmin,0.1,1)[0], \
                    numpy.random.normal(xmin,0.1,1)[0], \
                    numpy.random.uniform(0.01,1.0,1)[0], \
                    numpy.random.uniform(0,numpy.pi/2,1)[0], \
                    numpy.random.uniform(0,numpy.pi,1)[0], \
                    numpy.random.uniform(0.0001,1.0,1)[0]])
    elif funct[0] == 'ring':
        ndim, nwalkers = 6, 250

        pos = []
        for i in range(nwalkers):
            r_min = numpy.random.uniform(0.01,1.0,1)[0]

            pos.append([numpy.random.normal(xmin,0.1,1)[0], \
                    numpy.random.normal(xmin,0.1,1)[0], \
                    r_min, \
                    numpy.random.uniform(r_min,1.0,1)[0], \
                    numpy.random.uniform(0,numpy.pi/2,1)[0], \
                    numpy.random.uniform(0,numpy.pi,1)[0], \
                    numpy.random.uniform(0.0001,1.0,1)[0]])

    # Set up the MCMC simulation.

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
            args=(x, y, z, zerr))

    # Run a frew burner steps.

    pos, prob, state = sampler.run_mcmc(pos, 50)

    for i in range(ndim):
        fig, ax = plt.subplots(nrows=1, ncols=1)

        for j in range(nwalkers):
            ax.plot(sampler.chain[j,:,i])

        plt.show()

        plt.close(fig)

    sampler.reset()

    # Run the real MCMC simulation.

    sampler.run_mcmc(pos, 100)

    # Get the best fit parameters and uncertainties.

    samples = sampler.chain.reshape((-1, ndim))

    params = samples.mean(axis=0)
    sigma = samples.std(axis=0)

    # Clip a few stragglers and re-calculate.

    for k in range(2):
        good = numpy.repeat(True, samples.shape[0])
        print(samples.shape)
        if k == 0:
            nsigma=3.
        else:
            nsigma=4.

        for i in range(good.size):
            for j in range(ndim):
                if abs(samples[i,j] - params[j]) > nsigma*sigma[j]:
                    good[i] = False
                    break

        samples = samples[good,:]

        params = samples.mean(axis=0)
        sigma = samples.std(axis=0)
    print(samples.shape)

    return params, sigma, samples
