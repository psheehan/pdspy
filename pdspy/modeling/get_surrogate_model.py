import torch
import gpytorch

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy, \
        IndependentMultitaskVariationalStrategy

import pickle
import numpy
import time
import os

def get_surrogate_model(params, model="pringle+ulrich+diana", \
        quantity="temperature", nthreads=1):

    fitdir = os.path.dirname(os.path.abspath(__file__))+\
            "/surrogate_models/{0:s}/{1:s}".format(model, quantity)

    # Load the keys for the parameters of the surrogate model.

    keys = list(numpy.loadtxt(fitdir+"/keys.txt", dtype=str))

    # Load in the PCA that was found.

    pca = pickle.load(open(fitdir+"/pca.pkl", "rb"))

    # Load in the Gaussian process fits.

    torch.load(fitdir+"/inducing_points.pt")

    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(\
            num_tasks=ncomponents, rank=0, has_global_noise=False)

    model.load_state_dict(torch.load(fitdir+'/gp.pth'))
    likelihood.load_state_dict(torch.load(fitdir+'/likelihood.pth'))

    # Load in the data.

    x = []
    x.append([(params[k] - -9.)**3 if k == "logM_env" else params[k] for k in \
            keys])
    x = numpy.array(x)
    shape = (99,100,1)

    # Reconstruct the data from the PCA + GP fit.

    with torch.no_grad():
        t1 = time.time()
        components = [numpy.concatenate((\
                model(torch.from_numpy(x).float()).sample().detach().\
                numpy()[0],\
                numpy.repeat(0., y_grid.shape[1] - ncomponents)))]
        projected = pca.inverse_transform(components)
        t2 = time.time()
        print("Time to reconstruct = {0:f} seconds".format(t2 - t1))

    # Return the projected quantity.

    return 10.**projected[0].reshape(shape)

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.\
                NaturalVariationalDistribution(inducing_points.size(-2), \
                batch_shape=torch.Size([ncomponents]))
        variational_strategy = IndependentMultitaskVariationalStrategy(\
                VariationalStrategy(self, inducing_points, \
                variational_distribution, learn_inducing_locations=True), \
                num_tasks=ncomponents)

        super(GPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=\
                torch.Size([ncomponents]))

        base_covar_module = gpytorch.kernels.ScaleKernel(\
                gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=len(keys), \
                active_dims=[0,1,2,3,4,5,6,7], \
                batch_shape=torch.Size([ncomponents])), \
                batch_shape=torch.Size([ncomponents]))

        self.covar_module = base_covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
