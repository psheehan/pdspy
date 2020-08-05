import schwimmbad
import pickle
import numpy
import time
import os

def get_surrogate_model(params, model="pringle+ulrich+diana", \
        quantity="temperature", nthreads=1):

    # Load the keys for the parameters of the surrogate model.

    keys = list(numpy.loadtxt(os.path.dirname(os.path.abspath(__file__))+\
            "/surrogate_models/{0:s}/{1:s}/keys.txt".format(model, quantity), \
            dtype=str))

    # Load in the PCA that was found.

    pca = pickle.load(open(os.path.dirname(os.path.abspath(__file__))+\
            "/surrogate_models/{0:s}/{1:s}/pca.pkl".format(model, quantity), \
            "rb"))

    # Also the transformed data

    y_grid = numpy.load(os.path.dirname(os.path.abspath(__file__))+\
            "/surrogate_models/{0:s}/{1:s}/transformed_data.npy".\
            format(model, quantity))

    # Also load in the Gaussian process fits.

    gps = pickle.load(open(os.path.dirname(os.path.abspath(__file__))+\
            "/surrogate_models/{0:s}/{1:s}/gps.pkl".format(model, quantity), \
            "rb"))

    # Load the samples and make sure the best fit values are used for the 
    # hyperparameters.

    ncomponents = y_grid.shape[1]
    ncomponents = 9

    samples = []
    for i in range(ncomponents):
        samples += [numpy.load(os.path.dirname(os.path.abspath(__file__))+\
                "/surrogate_models/{0:s}/{1:s}/gp_samples_component{2:d}.pkl."
                "npy".format(model, quantity, i))]

    # Load in the data.

    x = []
    data = []

    x.append([(params[k] - -9.)**3 if k == "logM_env" else params[k] for k in \
            keys])

    x = numpy.array(x)
    shape = (99,100,1)

    # Set the parameter vector for the GPs randomly from the posteriors.

    for i in range(ncomponents):
        w = numpy.random.randint(samples[i].shape[0])

        gps[i].set_parameter_vector(samples[i][w])

    # Reconstruct the data from the PCA + GP fit.

    sample = lambda i: gps[i].sample_conditional(y_grid[:,i], x)[0] if \
            i < ncomponents else 0.

    t1 = time.time()
    with schwimmbad.JoblibPool(nthreads) as pool:
        components = [list(pool.map(sample, range(15)))]

    projected = pca.inverse_transform(components)
    t2 = time.time()
    print("Time to reconstruct = {0:f} seconds".format(t2 - t1))

    # Return the projected quantity.

    return 10.**projected[0].reshape(shape)
