from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import torch
from gp_project_pytorch.model.experimental.GPLVM_simple import GPLVM

# Load the iris dataset
iris = datasets.load_iris()
data_iris = iris.data

# Standardize the data
scaler = StandardScaler()
data_iris_standardized = scaler.fit_transform(data_iris)

data_iris_standardized = torch.tensor(data_iris_standardized, dtype=torch.float32)

# Apply the GPLVM model to Iris dataset
gplvm_iris = GPLVM(data_iris_standardized, latent_dim=2)
gplvm_iris.optimize(num_epochs=1000, lr=1e-2)

# Extract the learned latent variables
latent_iris = gplvm_iris.X.detach().numpy()

latent_iris.shape