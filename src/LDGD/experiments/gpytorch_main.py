# Standard imports
import matplotlib.pylab as plt
import torch
import os
import numpy as np
from pathlib import Path
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
import urllib.request
import tarfile
from gpytorch.models.gplvm.latent_variable import *
from gp_project_pytorch.model import *
from gpytorch.likelihoods import GaussianLikelihood, SoftmaxLikelihood
from tqdm.notebook import trange
from tqdm import tqdm

# Setting manual seed for reproducibility
torch.manual_seed(73)
np.random.seed(73)

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)

# If you are running this notebook interactively
wdir = Path(os.path.abspath('')).parent.parent
os.chdir(wdir)

url = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
urllib.request.urlretrieve(url, '3PhData.tar.gz')
with tarfile.open('3PhData.tar.gz', 'r') as f:
    f.extract('DataTrn.txt')
    f.extract('DataTrnLbls.txt')

Y = torch.Tensor(np.loadtxt(fname='DataTrn.txt'))
labels = torch.Tensor(np.loadtxt(fname='DataTrnLbls.txt'))
labels = (labels @ np.diag([1, 2, 3])).sum(axis=1)

N = len(Y)
C = len(np.unique(labels))
D = Y.shape[-1]
data_dim = Y.shape[1]
latent_dim = data_dim
n_inducing = 25
pca = False

"==================================== Point Estimate ======================================"
# Initialize likelihood and model
likelihood = GaussianLikelihood()
train_x = torch.nn.Parameter(torch.randn(N, latent_dim))
model = GPLVM(train_x, Y, likelihood)

# Use the adam optimizer
optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': [train_x]}], lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 500
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, Y)
    loss.backward()
    optimizer.step()

