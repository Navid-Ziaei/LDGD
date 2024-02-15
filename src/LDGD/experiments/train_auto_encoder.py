import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from LDGD.model.variational_autoencoder import VAE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create a transofrm to apply to each datapoint
transform = transforms.Compose([transforms.ToTensor()])

# download the MNIST datasets
path = '~/datasets'
train_dataset = MNIST(path, transform=transform, download=True)

x, y = train_dataset.data / torch.max(train_dataset.data), train_dataset.targets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = VAE(input_dim=784, hidden_dim=400, latent_dim=2, num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = model.fit(x=x_train, y=y_train, optimizer=optimizer, epochs=100, batch_size=500)

x_hat, y_hat, mean, log_var = model.predict(x_test)

predicet_label = np.argmax(y_hat, axis=-1)
print(classification_report(y_true=y_test, y_pred=predicet_label))

plt.figure(figsize=(10, 10))
plt.scatter(mean[:, 0], mean[:, 1], c=y_test, cmap='rainbow')
plt.show()
