import numpy
import emcee
import matplotlib.pyplot as plt
import scipy.signal
from .model import model

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

    x = numpy.arange(-10,10,0.1)
    y = numpy.arange(-10,10,0.1)

    params = numpy.array([])

    xmin = []
    ymin = []

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
    
        xmin.append(x[numpy.where(chisq == chisq.min())[0][0]])
        ymin.append(y[numpy.where(chisq == chisq.min())[1][0]])

        params = numpy.concatenate((params, [xmin[-1], ymin[-1], flux0]))

    # Next do a few iterations of MCMC to get the correct solution.

    x = data.u
    y = data.v
    z = numpy.concatenate((data.real, data.imag))[:,0]
    zerr = 1./numpy.sqrt(numpy.concatenate((data.weights,data.weights)))[:,0]
    print(xmin[0], ymin[0])

    # Define a likelihood function.

    pa0 = [numpy.pi for i in range(len(funct))]
    pa_range = [numpy.pi for i in range(len(funct))]

    for count in range(2):
        def lnlike(p, x, y, z, zerr, funct):
            m = model(x, y, p, return_type="append", funct=funct)

            return -0.5*(numpy.sum((z - m)**2 / zerr**2))

        def lnprior(p, funct):
            nparams = numpy.zeros(funct.size)
            for i in range(funct.size):
                if funct[i] == "point":
                    nparams[i] = 3
                elif funct[i] == "gauss":
                    nparams[i] = 6
                elif funct[i] == "circle":
                    nparams[i] = 6
                elif funct[i] == "ring":
                    nparams[i] = 7

            ind = 0

            for i in range(funct.size):
                if funct[i] == 'point':
                    if 0. < p[int(ind+2)]:
                        continue
                    else:
                        return -numpy.inf
                elif funct[i] == 'gauss':
                    if 0 < p[int(ind+2)] and 0 < p[int(ind+3)] <= p[int(ind+2)]\
                            and pa0[i]-pa_range[i]<= p[int(ind+4)] <= \
                            pa0[i]+pa_range[i] and 0. < p[int(ind+5)]:
                        continue
                    else:
                        return -numpy.inf
                elif funct[i] == 'circle':
                    if 0. < p[int(ind+2)] and 0. <= p[int(ind+3)] <= numpy.pi/2\
                            and pa0[i]-pa_range[i] <= p[int(ind+4)] <= \
                            pa0[i]+pa_range[i] and 0. < p[int(ind+5)]:
                        continue
                    else:
                        return -numpy.inf
                elif funct[i] == 'ring':
                    if 0. < p[int(ind+2)] and p[int(ind+2)] < p[int(ind+3)] and\
                            0.0 <= p[int(ind+4)] <= numpy.pi/2 and \
                            pa0[i]-pa_range[i] <= p[int(ind+5)] <= \
                            pa0[i]+pa_range[i] and 0. < p[int(ind+6)]:
                        continue
                    else:
                        return -numpy.inf

                ind += nparams[i]

            return 0.

        def lnprob(p, x, y, z, zerr, funct):
            lp = lnprior(p, funct)

            if not numpy.isfinite(lp):
                return -numpy.inf

            return lp + lnlike(p, x, y, z, zerr, funct)

        # Set up the emcee run.

        if count == 0:
            # Set the number of walkers and  dimensions.

            nwalkers = 250
            ndim = 0
            for i in range(len(funct)):
                if funct[i] == 'point':
                    ndim += 3
                elif funct[i] == 'gauss':
                    ndim += 6
                elif funct[i] == 'circle':
                    ndim += 6
                elif funct[i] == 'ring':
                    ndim += 7
            ndim = int(ndim)

            # Now set the positions of the walkers.

            pos = []
            for i in range(nwalkers):
                tmp_pos = []
                for j in range(len(funct)):
                    if funct[j] == 'point':
                        tmp_pos += [numpy.random.normal(xmin[j],0.1,1)[0], \
                                numpy.random.normal(ymin[j],0.1,1)[0], \
                                numpy.random.uniform(0.0001,1.0,1)[0]]
                    elif funct[j] == 'gauss':
                        sigma_x = numpy.random.uniform(0.01,10.0,1)[0]

                        tmp_pos += [numpy.random.normal(xmin[j],1.0e-4,1)[0], \
                                numpy.random.normal(ymin[j],1.0e-4,1)[0], \
                                sigma_x, \
                                numpy.random.uniform(0.01,sigma_x,1)[0], \
                                numpy.random.uniform(pa0-pa_range,\
                                pa0+pa_range,1)[0],\
                                numpy.random.normal(flux0,1.0e-4,1)[0]]
                    elif funct[j] == 'circle':
                        tmp_pos += [numpy.random.normal(xmin[j],1.0e-4,1)[0], \
                                numpy.random.normal(ymin[j],1.0e-4,1)[0], \
                                numpy.random.uniform(0.01,10.0,1)[0], \
                                numpy.random.uniform(0,numpy.pi/2,1)[0], \
                                numpy.random.uniform(pa0-pa_range,\
                                pa0+pa_range,1)[0], \
                                numpy.random.uniform(flux0,1.0e-4,1)[0]]
                    elif funct[j] == 'ring':
                        r_min = numpy.random.uniform(0.01,1.0,1)[0]

                        tmp_pos += [numpy.random.normal(xmin[j],0.1,1)[0], \
                                numpy.random.normal(ymin[j],0.1,1)[0], \
                                r_min, \
                                numpy.random.uniform(r_min,1.0,1)[0], \
                                numpy.random.uniform(0,numpy.pi/2,1)[0], \
                                numpy.random.uniform(pa0-pa_range,\
                                pa0+pa_range,1)[0], \
                                numpy.random.uniform(0.0001,1.0,1)[0]]

                pos.append(tmp_pos)

        # Set up the MCMC simulation.

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
                args=(x, y, z, zerr, funct))

        # Run a frew burner steps.

        if count == 0:
            pos, prob, state = sampler.run_mcmc(pos, 250)
        else:
            pos, prob, state = sampler.run_mcmc(pos, 50)

        # Make a plot of the steps.

        """
        for i in range(ndim):
            fig, ax = plt.subplots(nrows=1, ncols=1)

            for j in range(nwalkers):
                ax.plot(sampler.chain[j,:,i])

            plt.show()

            plt.close(fig)
        """

        # Because of the bimodality of p.a. we need to fix where the p.a. 
        # window is centered on.

        for i in range(len(funct)):
            if funct[i] == "point":
                pass
            else:
                if count == 0:
                    if funct[i] == "gauss" or funct[i] == "circle":
                        ind = 4 + sum([nparams[j] for j in range(i)])
                    elif funct[i] == "ring":
                        ind = 5 + sum([nparams[j] for j in range(i)])

                    samples = sampler.chain.reshape((-1,ndim))[:,ind]
                    pa_hist, bins = numpy.histogram(samples, 20)

                    bin_centers = (bins[0:-1] + bins[1:]) / 2.

                    extrema = scipy.signal.argrelextrema(pa_hist, \
                            numpy.greater)[0]
                    pa0[i] = bin_centers[extrema[0]]
                    e_current = extrema[0]
                    for  e in extrema[1:]:
                        if pa_hist[e] > pa_hist[e_current]:
                            pa0[i] = bin_centers[e]
                            e_current = e

                    pa0[i] = numpy.fmod(numpy.fmod(pa0[i], numpy.pi)+numpy.pi, \
                            numpy.pi)
                    pa_range[i] = numpy.pi/2

                    # Put all the walkers into the correct p.a. range.

                    for position in pos:
                        while position[ind] < pa0[i] - numpy.pi/2 or \
                                position[ind] >= pa0[i] + numpy.pi/2:
                            if position[ind] < pa0[i] - numpy.pi/2:
                                position[ind] += numpy.pi
                            elif position[ind] >= pa0[i] + numpy.pi/2:
                                position[ind] -= numpy.pi

        # Reset the sampler.

        sampler.reset()

    # Run the real MCMC simulation.

    sampler.run_mcmc(pos, 100)

    # Get the best fit parameters and uncertainties.

    samples = sampler.chain.reshape((-1, ndim))

    params = numpy.median(samples,axis=0)
    sigma = samples.std(axis=0)

    # Clip a few stragglers and re-calculate.

    for k in range(2):
        good = numpy.repeat(True, samples.shape[0])
        if k == 0:
            nsigma = 3.
        else:
            nsigma = 4.

        for i in range(good.size):
            for j in range(ndim):
                if abs(samples[i,j] - params[j]) > nsigma*sigma[j]:
                    good[i] = False
                    break

        samples = samples[good,:]

        params = numpy.median(samples, axis=0)
        sigma = samples.std(axis=0)

    # Clip a few stragglers and re-calculate.

    return params, sigma, samples
