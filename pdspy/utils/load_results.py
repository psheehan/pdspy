from sklearn.neighbors import KernelDensity
import scipy.signal
import scipy.stats
import emcee
import numpy

def load_results(config, model_path='', code="dynesty", discard=100, \
        best="median", unc="std", percentile=68, chisq_cut=False, \
        trim=None):
    # Get the list of parameters.

    keys = []
    for key in sorted(config.parameters.keys()):
        if not config.parameters[key]["fixed"]:
            keys.append(key)
    ndim = len(keys)

    # Load in the samples from the model fit.

    if code == "dynesty":
        samples = numpy.load(model_path+"samples.npy")
    elif code == "emcee":
        chain = numpy.load(model_path+"chain.npy")
        prob = numpy.load(model_path+"prob.npy")

        samples = chain[:,discard:,:].reshape((-1, ndim))
        prob = prob[:,discard:].reshape((-1,))
    elif code == "emcee3":
        backend = emcee.backends.HDFBackend("results.hdf5")

        samples = backend.get_chain(discard=discard, flat=True)
        prob = backend.get_log_prob(discard=discard, flat=True)

    # Make adjustments to the samples.

    if chisq_cut:
        good = 2 * (prob - prob.max()) > -scipy.stats.chi2.isf(q=0.01/\
                samples.shape[0], df=ndim)
    else:
        good = numpy.ones(samples.shape[0], dtype=bool)

    if trim != None:
        for command in trim:
            command = command.split(" ")

            for i, key in enumerate(keys):
                if key == command[0]:
                    if command[1] == '<':
                        good = numpy.logical_and(good, samples[:,i] > \
                                float(command[2]))
                    else:
                        good = numpy.logical_and(good, samples[:,i] < \
                                float(command[2]))

    samples = samples[good,:]

    # Get the best fit parameters.

    if best == "median":
        params = numpy.median(samples, axis=0)
    elif best == "peak":
        zscored_samples = (samples - samples.min(axis=0)) / \
            (samples.max(axis=0) - samples.min(axis=0))

        params = []
        for i in range(samples.shape[1]):
            bw = (samples.shape[0] * (1+2) / 4.)**(-1./(1+4)) / 3.
            kde = KernelDensity(kernel="gaussian", bandwidth=bw).\
                    fit(zscored_samples[:,i:i+1])

            xtest = numpy.linspace(samples[:,i].min(), samples[:,i].max(), 100)
            zscored_xtest = (xtest - samples[:,i].min()) / \
                    (samples[:,i].max() - samples[:,i].min())

            scores = kde.score_samples(zscored_xtest[:,None])

            params.append(xtest[numpy.argmax(scores)])

        params = numpy.array(params)
    elif best == "kde":
        zscored_samples = (samples - samples.min(axis=0)) / \
            (samples.max(axis=0) - samples.min(axis=0))

        bw = (samples.shape[0] * (ndim+2) / 4.)**(-1./(ndim+4))

        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(\
                zscored_samples)
        scores = kde.score_samples(zscored_samples)

        params = samples[numpy.argmax(scores)]

    # Get the uncertainties.

    if unc == "std":
        sigma = samples.std(axis=0)
    elif unc == "percentile":
        percentiles = numpy.percentile(samples, [50-percentile/2, \
                50+percentile/2], axis=0)

        sigma_up = percentiles[1] - params
        sigma_down = params - percentiles[0]

    # Make a dictionary of the best fit parameters.

    params = dict(zip(keys, params))

    return keys, params, samples
