#import matplotlib.pyplot as plt
from ..constants.astronomy import arcsec
import numpy
import scipy
import dynesty
import dynesty.utils as dyfunc
#import dynesty.plotting as dyplot
from .model import model

def fit_model(data, funct='point', nsteps=1e3, niter=3, max_size=numpy.inf, \
        xmax=10., ymax=10., step_size=0.1, min_separation=1., \
        primary_beam=None, image_rms=None, nlive=250):

    if type(funct) == str:
        funct = numpy.array([funct])
    elif type(funct) == list:
        funct = numpy.array(funct)
    elif type(funct) == numpy.ndarray:
        pass

    # Check if we need to adjust the data weights based on a supplied 
    # image_rms.

    if image_rms != None:
        sigma_tot = 1./numpy.sqrt(data.weights.sum())
        ratio = sigma_tot / image_rms

        data.weights *= ratio**2
    else:
        # If not, we could still use the image rms as a lower limit on flux
        # for the modeling.
        image_rms = 1./numpy.sqrt(data.weights.sum())

    # If the max_size == inf, calculate based on the data as dynesty needs a
    # finite range.

    if max_size == numpy.inf:
        max_size = 1./data.uvdist[data.uvdist > 0].min() / arcsec

    # Set up the data properly.

    good = data.weights[:,0] > 0
    x = data.u[good]
    y = data.v[good]
    z = numpy.concatenate((data.real[good,:], data.imag[good,:]))[:,0]
    zerr = 1./numpy.sqrt(numpy.concatenate((data.weights[good,:],data.weights[good,:])))[:,0]

    # Set up some parameters for dynesty.

    periodic = []
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

        if funct[i] in ['gauss','circle','ring']:
            periodic.append(int(nparams.sum()-2))

    ndim = int(nparams.sum())
    nlive = int(4*xmax*ymax / step_size**2)

    xlim = [[-xmax, xmax] for i in range(funct.size)]
    ylim = [[-ymax, ymax] for i in range(funct.size)]

    # Set up the dynesty sampler.

    sampler = dynesty.NestedSampler(lnlike, ptform, ndim=ndim, nlive=nlive, \
            bound='multi', logl_args=(x, y, z, zerr, funct, nparams,\
            primary_beam, data.freq.mean()), ptform_args=(funct, nparams, \
            xlim, ylim, max_size, min_separation, image_rms), periodic=periodic)

    # Run the nested sampling.

    sampler.run_nested(dlogz=0.05)

    # Get the samples.

    res = sampler.results
    samples, weights = res.samples, numpy.exp(res.logwt - res.logz[-1])
    samples = dyfunc.resample_equal(samples, weights)

    # Make a traceplot.

    """
    fig, axes = dyplot.traceplot(res, show_titles=True, trace_cmap='viridis', \
            connect=True, connect_highlight=range(5))

    plt.show(fig)
    """

    # Correct values that are in log.

    ind = 0
    for i in range(funct.size):
        if funct[i] in ['gauss','circle','ring']:
            samples[:,int(ind+2)] = 10.**samples[:,int(ind+2)]

            if funct[i] in ['gauss','ring']:
                samples[:,int(ind+3)] = 10.**samples[:,int(ind+3)]

        samples[:,int(ind+nparams[i]-1)] = 10.**samples[:,int(ind+nparams[i]-1)]

        ind += nparams[i]

    # Get the best fit parameters.

    params = numpy.median(samples, axis=0)
    sigma = samples.std(axis=0)

    # Clip a few stragglers and re-calculate.

    return params, sigma, samples

# Define a likelihood function.

def lnlike(p, x, y, z, zerr, funct, nparams, primary_beam, freq):
    params = p.copy()

    # p values are supplied in log-space for many parameters, so correct those.
    ind = 0
    for i in range(funct.size):
        if funct[i] in ['gauss','circle','ring']:
            params[int(ind+2)] = 10.**params[int(ind+2)]

            if funct[i] in ['gauss','ring']:
                params[int(ind+3)] = 10.**params[int(ind+3)]

        params[int(ind+nparams[i]-1)] = 10.**params[int(ind+nparams[i]-1)]

        ind += nparams[i]

    # Generate the model.
    m = model(x, y, params, return_type="append", funct=funct, \
            primary_beam=primary_beam, freq=freq)

    # Calculate the chi^2 value.
    return -0.5*(numpy.sum((z - m)**2 / zerr**2))

# Define the prior transformation function.

def ptform(u, funct, nparams, xlim, ylim, max_size, min_separation, image_rms):
    p = numpy.array(u)

    # Log of the maximum size.
    logr_max = numpy.log10(max_size)

    # Use 3*rms as minimum flux value allowed.
    logF_min = numpy.log10(3. * image_rms)

    ind = 0
    x0, y0 = [], []
    for i in range(funct.size):
        # Always sample x from the full range.
        p[int(ind+0)] = (xlim[i][1] - xlim[i][0])*u[int(ind+0)] + xlim[i][0]

        # For y, we need to sample from everywhere but the y-values that are
        # too close to a previous source, for a given x.
        if i == 0:
            p[int(ind+1)] = (ylim[i][1] - ylim[i][0])*u[int(ind+1)] + \
                    ylim[i][0]
        else:
            # Only use the slow generation function if the x value of a previous
            # component is too close to the x-value of this component.
            if numpy.all(numpy.abs(p[int(ind+0)] - numpy.array(x0)) > \
                    min_separation):
                p[int(ind+1)] = (ylim[i][1] - ylim[i][0])*u[int(ind+1)] + \
                        ylim[i][0]
            else:
                p[int(ind+1)] = y_gen(ylim[i][0], ylim[i][1], p[int(ind+0)], \
                        x0, y0, min_separation).ppf(u[int(ind+1)])

        # Update our list of sources to watch out for.
        x0.append(p[int(ind+0)])
        y0.append(p[int(ind+1)])

        # Set ranges for models that have a size.
        if funct[i] in ['gauss','circle','ring']:
            p[int(ind+2)] = (logr_max - -3.)*u[int(ind+2)] - 3.

            if funct[i] in ['gauss','ring']:
                p[int(ind+3)] = (p[int(ind+2)] - -3.)*u[int(ind+3)] - 3.

            # Set the inclination for models that use it.
            if funct[i] == 'circle':
                p[int(ind+3)] = numpy.pi/2 * u[int(ind+3)]
            elif funct[i] == 'ring':
                p[int(ind+4)] = numpy.pi/2 * u[int(ind+4)]

            # Set the position angle.
            p[int(ind+nparams[i]-2)] = numpy.pi*u[int(ind+nparams[i]-2)]

        # Each source has to be less bright than the last.
        if i == 0:
            logF_max = 0.
        else:
            logF_max = p[int(ind-1)]

        p[int(ind+nparams[i]-1)] = (logF_max - logF_min)*\
                u[int(ind+nparams[i]-1)] + logF_min

        ind += nparams[i]

    return p

# Functions for helping to sample multiple systems.

def func(y, x, x0, y0, radius):
    if type(x0) == list:
        x0 = numpy.array(x0)
    if type(y0) == list:
        y0 = numpy.array(y0)

    return numpy.maximum(0, 1 - numpy.exp(-(((x - x0)**2 + (y - y0)**2) / \
            radius**2)**4).sum())

class y_gen(scipy.stats.rv_continuous):
    def __init__(self, a=None, b=None, x=None, x0=None, y0=None, radius=None):
        self.x = x
        self.x0 = x0
        self.y0 = y0
        self.radius = radius
        
        self.norm = scipy.integrate.quad(func, a, b, args=(x, x0, y0, \
                radius))[0]

        super().__init__(a=a, b=b)

    def _pdf(self, y):
        return func(y, self.x, self.x0, self.y0, self.radius) / self.norm
