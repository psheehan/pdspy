#!/usr/bin/env python3

import argparse
import numpy
import os
import pyDOE
import sys

import pdspy.modeling as modeling
import pdspy.utils as utils

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle
import glob

import torch
import gpytorch

from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy, \
        IndependentMultitaskVariationalStrategy

import astropy.stats

################################################################################
#
# Command line arguments.
#
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', type=int, default=0)
parser.add_argument('-t', '--total_array_tasks', type=int, default=1)
parser.add_argument('-a', '--action')
parser.add_argument('-n', '--ncpus', type=int, default=1)
parser.add_argument('-l', '--nlearn', type=int, default=1)
parser.add_argument('-m', '--nlearn_minor', type=int, default=1)
parser.add_argument('-g', '--grid', type=str, default="train")
args = parser.parse_args()

################################################################################
#
# Set up the parameters of the grid.
#
################################################################################

config = utils.load_config()

keys = []
for key in config.parameters:
    if not config.parameters[key]["fixed"]:
        keys.append(key)

################################################################################
#
# Create a function to create a set of samples of parameter space.
#
################################################################################

def generate_samples(nsamples, mode="LHS", grid_keys=keys):
    # Create the grid of parameters for models using a latin hypercube sampling:

    if mode == "LHS":
        samples = pyDOE.lhs(len(grid_keys), samples=nsamples, \
                criterion="maximin")
    else:
        samples = numpy.random.uniform(size=(nsamples,len(keys)))

    # Stretch the samples to the appropriate range.

    for i, key in enumerate(grid_keys):
        #limits = grid_params[key+"_range"]
        limits = config.parameters[key]["limits"]
        samples[:,i] = (limits[1] - limits[0])*samples[:,i] + limits[0]

    # Now turn this into an array that we can save.

    pars = []

    for i in range(nsamples):
        #pars.append(dict(zip(keys, samples[i,:])))

        pars.append(dict(zip(keys, [samples[i,\
                numpy.where(numpy.array(grid_keys) == par)[0][0]] if par in \
                grid_keys else config.parameters[par]["value"] for par in \
                keys])))

        # Now get the filename.

        pars[-1]["filename"] = "".join([key.replace("_","")+"_"+ \
                "{0:.5f}".format(pars[-1][key]).replace(".hdf5","").\
                #replace("pollack_", "").replace("88","").replace("66","")+ \
                replace("pollack_", "")+ \
                "_" for key in keys]).rstrip("_")+".hdf5"

    print(len(pars))
    parameters = numpy.array(pars)

    return parameters, samples

################################################################################
#
# Convenience function for getting the proper model directory.
#
################################################################################

def get_directory(grid="train", key=None):
    directory = grid.capitalize().replace("d","D") + "Grid" + \
            ("s" if "1" in grid else "") + "/"

    if key is not None:
        directory = directory + key + "/"

    return directory

################################################################################
#
# Create a function to run models from parameters.npy
#
################################################################################

def run_models(grid="train", key=None):
    # Get the directory to load from.

    directory = get_directory(grid=grid, key=key)

    # Load the parameters file.

    parameters = numpy.load(directory+"parameters.npy", allow_pickle=True)
    print(parameters.size)

    # Load the list of models that have been tried.

    tried_models = numpy.loadtxt(directory+"tried_models.txt", dtype=str)

    # Run the model:

    start = args.index
    step = args.total_array_tasks

    bad = []

    for i in range(start,parameters.size,step):
        base_dir = os.environ["PWD"]

        print(parameters[i])

        if i >= len(parameters):
            continue
        elif os.path.exists(directory+parameters[i]["filename"]):
            continue
        elif parameters[i]["filename"] in tried_models:
            continue

        if not os.path.exists(directory):
            os.mkdir(directory)

        try:
            model = modeling.run_disk_model(config.visibilities, config.images,\
                    config.spectra, parameters[i], config.parameters, \
                    timelimit=900, with_hyperion=True, \
                    percentile=config.percentile, absolute=config.absolute, \
                    relative=config.relative, ncpus=args.ncpus, \
                    ncpus_highmass=args.ncpus, verbose=True, \
                    increase_photons_until_convergence=True)

            # Write out the file.

            model.write_yso(directory+parameters[i]["filename"])
        except:
            f = open(directory+"tried_models.txt","a")
            f.write(parameters[i]["filename"]+"\n")
            f.close()
            #bad.append(i)

        """
        new_parameters = numpy.delete(parameters, bad)
        numpy.save(directory+"parameters.npy", new_parameters)
        """

################################################################################
#
# Load in the model grid.
#
################################################################################

def load_data(grid="train", key=None, pca=False):
    # Get the directory to load from.

    directory = get_directory(grid=grid, key=key)

    # Load in the parameters file with all of the model info.

    parameters = numpy.load(directory+"parameters.npy", allow_pickle=True)

    # Check whether a data file already exists.

    if os.path.exists(directory+"data.npz"):
        x = numpy.load(directory+"data.npz")['x'].tolist()
        data = numpy.load(directory+"data.npz")['data'].tolist()
        original = numpy.load(directory+"data.npz")['original'].tolist()
        filenames = numpy.load(directory+"data.npz")['filenames'].tolist()
        shape = tuple(numpy.load(directory+"data.npz")['shape'])

        start_index = len(data)
    else:
        data = []
        original = []
        x = []
        filenames = []

        start_index = 0

    if len(glob.glob(directory+"*.hdf5")) > len(data):
        for p in parameters[start_index:]:
            try:
                m = modeling.YSOModel()
                m.read_yso(directory+p["filename"])

                temperature = numpy.array(m.grid.temperature)
                density = numpy.array(m.grid.density)

                density[density == 0] = 1.0e-30

                original.append(temperature)

                data.append(((temperature[:,:,:,0]*density[:,:,:,0]).\
                        sum(axis=0) / density[:,:,:,0].sum(axis=0)).flatten())

                x.append([p[key] for key in keys])
                filenames.append(p["filename"])
            except:
                pass

    x = numpy.array(x)
    data = numpy.array(data)
    original = numpy.array(original)
    #shape = (99, 100, 1)
    try:
        shape = temperature[0].shape
    except:
        pass

    numpy.savez(directory+"data.npz", data=data, x=x, filenames=filenames, \
            original=original, shape=shape)

    data[data <= 0] = 0.1
    data = numpy.log10(data)

    if pca:
        # Load in the PCA that was found.

        pca = pickle.load(open("pca.pkl","rb"))

        # Load in the data.

        y = pca.transform(data)
        ncomponents = y.shape[1]

        return x, y, ncomponents
    else:
        return x, data, shape

################################################################################
#
# Create a function to do the PCA decomposition.
#
################################################################################

def run_pca_decomposition(plot=False):
    # First, load in the data.

    x, data, shape = load_data(grid="train", pca=False)
    print(data.shape)

    # Make some plots, if requested.

    if plot:
        # Compute a PCA decomposition.

        pca = PCA(n_components=100)
        pca.fit(data)

        # Plot the variance explained.

        fig, ax = plt.subplots(nrows=1, ncols=1, gridspec_kw=dict(left=0.2, \
                right=0.96, top=0.95, bottom=0.15))

        ax.plot(numpy.cumsum(pca.explained_variance_ratio_)*100)

        ax.set_xlabel("Number of PCA Components", fontsize=14, labelpad=10)
        ax.set_ylabel("Cumulative Variance Explained (\%)", fontsize=14, \
                labelpad=10)

        ax.tick_params(labelsize=14)

        ax.set_xlim(0,100)
        ax.set_ylim(80.,100.)

        #plt.show()
        plt.savefig("variance.png")

        # Get the eigentempertemperatures.

        fig, axes = plt.subplots(3, 5, figsize=(12,7), subplot_kw={"xticks":[],\
                "yticks":[], 'projection':'polar'}, gridspec_kw=dict(left=0.03,\
                right=0.97, top=0.95, bottom=0.02, hspace=0.3, wspace=0.15))

        r, theta = numpy.meshgrid(numpy.arange(shape[0]), numpy.linspace(0., \
                numpy.pi/2, shape[1])[::-1], indexing='ij')

        for i, ax in enumerate(axes.flat):
            ax.pcolor(theta, r, pca.components_[i].reshape(shape)[:,:,0], \
                    shading="auto")

            ax.set_title("Eigentemperature, $T_{{{0:d}}}$".format((i+1)))

            ax.set_thetalim(0.,numpy.pi/2)

            ax.set_rticks([])
            ax.set_xticks([])

        plt.savefig("eigentemperatures.png")

        # Now plot the reconstrcuted temperature for the first 9 
        # eigentemperatures.

        nrows = 16

        fig, ax = plt.subplots(2, nrows, figsize=(17.5,2.5), gridspec_kw=dict(\
                hspace=0.02,wspace=0.1, left=0.03, right=0.99, bottom=0.015, \
                top=0.92),\
                subplot_kw={'xticks':[], 'yticks':[], 'projection':'polar'})

        index = numpy.random.randint(data.shape[0])
        print("Plotting reconstruction for parameters:")
        print(dict(zip(keys, x[index,:])))

        for i in range(nrows):
            if i == nrows-1:
                ax[0,i].pcolor(theta, r, 10.**data[index].reshape(shape)\
                        [:,:,0], shading="auto")
                ax[1,i].set_axis_off()

                ax[0,i].set_rticks([])
                ax[0,i].set_xticks([])
                ax[1,i].set_rticks([])
                ax[1,i].set_xticks([])

                ax[0,i].set_thetalim(0., numpy.pi/2)

                ax[0,i].set_title("Actual")
            else:
                pca = PCA(n_components=int(3*i+1)).fit(data)
                components = pca.transform(data[index+0:index+1,:])
                projected = pca.inverse_transform(components)

                ax[0,i].pcolor(theta, r, 10.**projected[0].reshape(shape)\
                        [:,:,0], vmin=10.**data[index].min(), \
                        vmax=10.**data[index].max(), shading="auto")

                diff = 10.**projected[0] - 10.**data[index]

                ax[1,i].pcolor(theta, r, diff.reshape(shape)[:,:,0], \
                        vmin=-numpy.abs(diff).max(), vmax=numpy.abs(diff).\
                        max(), shading="auto")

                ax[0,i].set_title("N = {0:d}".format(int(3*i+1)))

                print("Maximum Error for {1:d} Components = {0:4.2f} K".format(\
                        numpy.abs(diff).max(), int(3*i+1)))

                ax[0,i].set_rticks([])
                ax[0,i].set_xticks([])
                ax[1,i].set_rticks([])
                ax[1,i].set_xticks([])

                ax[0,i].set_thetalim(0., numpy.pi/2)
                ax[1,i].set_thetalim(0., numpy.pi/2)

        ax[0,0].set_ylabel("Reconstructed\nTemperature")
        ax[1,0].set_ylabel("Reconstructed\n- Actual")

        #plt.show()
        plt.savefig("pca_reconstruction.png")

    # Now that we know how many features to use, generate the final PCA and 
    # save it.

    pca = PCA(n_components=50).fit(data)

    pickle.dump(pca, (open("pca.pkl","wb")))

################################################################################
#
# Create a Gaussian Process model to be used.
#
################################################################################

class GPModel(ApproximateGP):
    def __init__(self, inducing_points=None, ncomponents=1, load=False):
        if load:
            inducing_points = torch.load("inducing_points.pt")

            pca = pickle.load(open("pca.pkl","rb"))
            ncomponents = pca.n_components

        self.ncomponents = ncomponents

        variational_distribution = gpytorch.variational.\
                NaturalVariationalDistribution(inducing_points.size(-2), \
                batch_shape=torch.Size([ncomponents]))
        variational_strategy = IndependentMultitaskVariationalStrategy(\
                VariationalStrategy(self,inducing_points, \
                variational_distribution, learn_inducing_locations=True), \
                num_tasks=ncomponents)

        super(GPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=\
                torch.Size([ncomponents]))

        # Set up the GPs.

        self.covar_module = gpytorch.kernels.ScaleKernel(\
                gpytorch.kernels.MaternKernel(nu=1.5, \
                ard_num_dims=len(keys), \
                active_dims=list(range(len(keys))), \
                batch_shape=torch.Size([ncomponents])), \
                batch_shape=torch.Size([ncomponents]))

        # Set initial values.

        if load:
            self.load_state_dict(torch.load('gp.pth'))
            self.eval()
        else:
            self.covar_module.base_kernel.lengthscale = torch.reshape(\
                    torch.tensor([[[(config.parameters[k]["limits"][1] - \
                    config.parameters[k]["limits"][0])/5 \
                    for k in keys]] for i in range(ncomponents)]).float(),\
                    [ncomponents, 1, len(keys)])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

################################################################################
#
# Create a function to do the Gaussian Process fit.
#
################################################################################

def run_gp_fit():
    # Load in the data.

    x, y, ncomponents = load_data(grid="train", pca=True)

    # Set up the data as tensors.

    train_x = torch.from_numpy(x).float()
    train_y = torch.from_numpy(y[:,:ncomponents]).float()
    print(train_y.size(-1), ncomponents)

    # Create the Gaussian process.

    induce = numpy.random.choice(numpy.arange(train_x.size(0)), \
            size=min(500,train_x.size(0)), replace=False)

    inducing_points = train_x[induce, :].unsqueeze(0).repeat(ncomponents, 1, 1)

    model = GPModel(inducing_points=inducing_points, ncomponents=ncomponents)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(\
            num_tasks=ncomponents, rank=0, has_global_noise=False, \
            noise_constraint=gpytorch.constraints.GreaterThan(1e-3**2))

    likelihood.task_noises = 0.1**2

    # If a GPU is available, put the data there.

    if torch.cuda.is_available():
        inducing_points = inducing_points.cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

    # Find optimal hyperparameters

    model.train()
    likelihood.train()

    # Use the adam optimizer

    hyperparameter_optimizer = torch.optim.Adam([\
            {'params': model.hyperparameters()},\
            {'params': likelihood.parameters()}], lr=1.0e-2)

    variational_ngd_optimizer = gpytorch.optim.NGD(\
            model.variational_parameters(), num_data=train_y.size(0), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, \
            num_data=train_y.size(0))

    num_epochs = 3000

    parameters = []
    for i in range(num_epochs):
        for x_batch, y_batch in train_loader:
            variational_ngd_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()

            output = model(x_batch)

            loss = -mll(output, y_batch)
            loss.backward()

            variational_ngd_optimizer.step()
            hyperparameter_optimizer.step()

        print('Iter %d/%d - Loss: %.3f' % (i + 1, num_epochs, loss.item()))

        """
        # Store the sequence of steps for each parameter.

        labels = ["Loss"] + ["lengthscale_{0:s}".format(key) for key in \
                keys] + ["outputscale","constant","noise"]

        for icomp in range(ncomponents):
            parameters.append([loss.item(), *tuple(model.covar_module.\
                    base_kernel.lengthscale[icomp].detach().cpu().numpy().\
                    flatten()), \
                    model.covar_module.outputscale[icomp].cpu().item()**0.5, \
                    model.mean_module.constant[icomp].cpu().item(), \
                    likelihood.task_noises[icomp].cpu().item()**0.5])

        # Every 500 steps make a plot of how the parameters have been 
        # changing.
        if (i + 1) % 3000 == 0:
            for icomp in range(ncomponents):
                fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(12.8,9.6))

                for j, ax in enumerate(axes.flatten()):
                    if j >= numpy.array(parameters).shape[1]:
                        ax.set_axis_off()
                        continue

                    ax.plot(numpy.arange(i+1), numpy.array(parameters)\
                            [icomp::ncomponents,j])

                    ax.set_xlabel("Steps")
                    ax.set_ylabel(labels[j].replace("_",""))

                fig.tight_layout()

                fig.savefig("stepsplot{0:d}.png".\
                        format(icomp))
                plt.close(fig)
                plt.clf()
        """

    # Bring the data and models back from the GPU, if they were there.

    if torch.cuda.is_available():
        inducing_points = inducing_points.cpu()
        train_x = train_x.cpu()
        train_y = train_y.cpu()
        model = model.cpu()
        likelihood = likelihood.cpu()

    # Finally, save the model.

    torch.save(model.state_dict(), "gp.pth")
    torch.save(likelihood.state_dict(), "likelihood.pth")
    torch.save(inducing_points, 'inducing_points.pt')

################################################################################
#
# Test the GP fit against 1D slices.
#
################################################################################

def test_in_1d():
    # Directory to read results from.

    fitdir = "gpfit_plots"

    if not os.path.exists(fitdir):
        os.mkdir(fitdir)

    # Load in the data all at once and store it.

    x = {}
    y = {}

    for i, key in enumerate(keys):
        # Load in the data.

        x[key], y[key], ncomponents = load_data(grid="1d", key=key, pca=True)

        #y[key] = y[key][numpy.argsort(x[key][:,i]),:]
        #x[key] = numpy.sort(x[key][:,i])
        x[key] = x[key][:,i]

    # Also the transformed data

    model = GPModel(load=True)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(\
            num_tasks=ncomponents, rank=0, has_global_noise=False)
    likelihood.load_state_dict(torch.load('likelihood.pth'))
    likelihood.eval()

    # Loop through the different parameters and plot.

    for icomp in range(ncomponents):
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(9,4), \
                gridspec_kw=dict(left=0.06,right=0.95, top=0.95, bottom=0.1, \
                wspace=0.4, hspace=0.25))

        for ikey, (key, ax) in enumerate(zip(keys, axes.flatten())):
            # Generate a predicted curve and uncertainties.

            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # The x-values at which to predict values from the Gaussian 
                # processes.
                x_pred = torch.from_numpy(numpy.concatenate([\
                        numpy.linspace(config.parameters[k]["limits"][0], \
                        config.parameters[k]["limits"][1], 500).\
                        reshape((-1,1)) if k == key else \
                        numpy.repeat(config.parameters[k]["value"], 500).\
                        reshape((-1,1)) for k in keys], axis=1)).float()

                observed_pred = likelihood(model(x_pred))

                lower, upper = observed_pred.confidence_region()

            # Now make a plot of the best fit Gaussian process

            ax.errorbar(x[key], y[key][:,icomp], fmt=".k", capsize=0)

            ax.plot(x_pred[:,ikey].numpy(), observed_pred.mean[:,icomp].\
                    numpy(), 'k', alpha=0.5)
            ax.fill_between(x_pred[:,ikey].numpy(), lower[:,icomp].numpy(), \
                    upper[:,icomp].numpy(), color="k", alpha=0.25)

            ax.set_xlim(config.parameters[key]["limits"][0], \
                    config.parameters[key]["limits"][1])

            ax.set_xlabel("$"+key+"$")
            ax.set_ylabel("$w_{0:d}$".format(icomp+1))

        fig.savefig(fitdir+"/component{0:d}_1Dproj.png".format(icomp))

################################################################################
#
# Test the GP fit against the full parameter space test data.
#
################################################################################

def test_gp_full(plot=False):
    # Also the transformed data

    x_grid, y_grid, ncomponents = load_data(grid="test", pca=True)
    residuals = y_grid.copy()

    # Also load in the Gaussian process fits.

    model = GPModel(load=True)

    # Subtract off the mean from the original data.

    test_x = torch.from_numpy(x_grid).float()
    test_y = torch.from_numpy(y_grid).float()

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    with torch.no_grad():
        index = 0
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            means = preds.mean

            residuals[index:index+1000,:model.ncomponents] -= means.numpy()

            index += 1000

    icomp = 0

    print(residuals[:,icomp].min(), residuals[:,icomp].max(), \
            residuals[:,icomp].std(), astropy.stats.mad_std(residuals[:,icomp]))
    print(y_grid[:,icomp].max() - y_grid[:,icomp].min(), numpy.abs(\
            residuals[:,icomp]).max() / (y_grid[:,icomp].max() - \
            y_grid[:,icomp].min()))

    print(numpy.argmax(residuals[:,icomp]), \
            x_grid[numpy.argmax(residuals[:,icomp]),:])
    print(numpy.argmin(residuals[:,icomp]), \
            x_grid[numpy.argmin(residuals[:,icomp]),:])

    if plot:
        # Create a figure to put the results in.

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # Now plot the residuals.

        ax.scatter3D(x_grid[:,0], x_grid[:,1], x_grid[:,2], \
                c=residuals[:,icomp])

        plt.show()

################################################################################
#
# Predict whether a model will finish in the allowed time.
#
################################################################################

def predict_model_success(x_test, fail_thresh=0.95, grid="train"):
    # Get the directory to load from.

    directory = get_directory(grid=grid)

    # Load the parameters file.

    parameters = numpy.load(directory+"parameters.npy", allow_pickle=True)

    # Load the list of models that have been tried.

    tried_models = numpy.loadtxt(directory+"tried_models.txt", dtype=str)

    # Get the list of which models were successful and not.

    good = numpy.repeat(False, parameters.size)
    keep = numpy.repeat(True, parameters.size)
    for i in range(parameters.size):
        if os.path.exists(directory+parameters[i]["filename"]):
            good[i] = True
        elif parameters[i]["filename"] in tried_models:
            good[i] = False
        else:
            #print("It appears you have models that have not been run. Please finish making your grid before moving on.")
            #sys.exit(0)
            keep[i] = False

    good = good[keep]
    parameters = parameters[keep]

    x_train = numpy.array([list(parameters[i].values())[0:-1] for i in \
            range(parameters.size)])

    # Train the logistic regression model.

    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_train, good)

    # Evaluate the model on the data points in question.

    prediction = lr.predict(x_test)
    proba = lr.predict_proba(x_test)
    decision = lr.decision_function(x_test)

    return proba[:,0] < fail_thresh

################################################################################
#
# Learn the next point at which to fill parameter space with.
#
################################################################################

def learn_next_point(nlearn=1):
    # Also load in the Gaussian process fits.

    model = GPModel(load=True)

    # Subtract off the mean from the original data.

    parameters, x_grid = generate_samples(10000, mode="random")
    y_grid = numpy.random.uniform(size=(10000,model.ncomponents))

    # Only take the points that are likely to finish within the 15 minute 
    # timeframe.

    good = predict_model_success(x_grid, grid="train")

    parameters = parameters[good]
    x_grid = x_grid[good]
    y_grid = y_grid[good]

    # Use the GP to predict the variance and get the next point.

    test_x = torch.from_numpy(x_grid).float()
    test_y = torch.from_numpy(y_grid).float()
    variance = torch.from_numpy(y_grid).float()

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    with torch.no_grad():
        index = 0
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)

            variance[index:index+1000,:model.ncomponents] = preds.variance

            index += 1000

    # Now plot the residuals.

    cum = torch.cumsum(variance.sum(axis=1), 0) / variance.sum()
    max_variance_index = numpy.array([torch.argmax(cum[cum - numpy.random.uniform() < 0]) \
            for i in range(nlearn)])

    return parameters[numpy.array(numpy.unique(max_variance_index))]

################################################################################
#
# Here's where the code can take actions.
#
################################################################################

if (args.action == 'make'):
    grid = get_directory(grid=args.grid)

    if not os.path.exists(grid):
        os.mkdir(grid)

    os.chdir(grid)

    if grid == "1DGrids/":
        for key in keys:
            if not os.path.exists(key):
                os.mkdir(key)

            os.chdir(key)

            parameters, samples = generate_samples(config.nsamples_1d, \
                    mode="LHS", grid_keys=[key])
            numpy.save("parameters.npy", parameters)

            os.chdir("..")
    elif grid == "TestGrid/":
        parameters, samples = generate_samples(config.nsamples_test, \
                mode="random")
        numpy.save("parameters.npy", parameters)
    else:
        parameters, samples = generate_samples(config.nsamples_train, \
                mode="LHS")
        numpy.save("parameters.npy", parameters)

    os.chdir("..")

elif (args.action == 'run'):
    if "1" in args.grid:
        for key in keys:
            run_models(grid=args.grid, key=key)
    else:
        run_models(grid=args.grid)

elif (args.action == 'pca'):
    run_pca_decomposition(plot=True)

elif (args.action == 'gpfit'):
    run_gp_fit()

elif (args.action == "test"):
    if "1" in args.grid:
        test_in_1d()
    elif args.grid == "test":
        test_gp_full(plot=True)

elif (args.action == 'learn'):
    directory = get_directory(grid="train")

    run_pca_decomposition(plot=False)
    run_gp_fit()

    for i in range(args.nlearn // args.nlearn_minor):
        # Do stuff

        new_parameters = learn_next_point(nlearn=args.nlearn_minor)
        print(new_parameters)

        # Update the parameters.

        old_parameters = numpy.load(directory+"parameters.npy", \
                allow_pickle=True)

        parameters = numpy.concatenate((old_parameters, new_parameters))

        numpy.save(directory+"parameters.npy", parameters)

        # Run the new model to add it to the grid.

        run_models(grid="train")

        # Re-run the PCA and GP fits to update the surrogate model.

        run_pca_decomposition(plot=False)
        run_gp_fit()

elif (args.action == "finish"):
    directory = get_directory(grid="test")

    parameters = numpy.load(directory+"parameters.npy", allow_pickle=True)

    x_test = numpy.array([list(parameters[i].values())[0:-1] for i in \
            range(parameters.size)])

    good = predict_model_success(x_test, grid="train")
    print(good.sum())

    for i in range(parameters.size):
        if not good[i]:
            for key in parameters[i]:
                print(key, parameters[i][key])
            print()
