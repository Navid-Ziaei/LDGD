import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

class VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, num_classes=2):
        super(VAE, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        ).to(self.device)

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder_reg = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        ).to(self.device)

        self.decoder_cls = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)
        ).to(self.device)

    def encode(self, x):
        x = self.encoder(x)
        mean, log_var = self.mean_layer(x), self.logvar_layer(x)
        return mean, log_var

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def decode_reg(self, x):
        return self.decoder_reg(x)

    def decode_cls(self, x):
        return self.decoder_cls(x)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode_reg(z)
        y_hat = self.decode_cls(z)
        return x_hat, y_hat, mean, log_var

    def loss_function(self, x, y, x_hat, y_hat, mean, log_var):
        reconstruction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
        classification_loss = nn.functional.cross_entropy(y_hat, y, reduction='sum')
        KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reconstruction_loss, classification_loss, KLD

    def fit(self, x, y, x_test, y_test, optimizer, epochs, batch_size=100):
        self.train()
        # Convert the input data to a TensorDataset and DataLoader for batch processing
        dataset = TensorDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        history = {
            'loss': [],
            'loss_rec': [],
            'loss_cls_list': [],
            'loss_kl_list': [],
            'accuracy_train': [],
            'accuracy_test': []
        }
        loss_list, loss_rec_list, loss_cls_list, loss_kl_list = [], [], [], []
        for epoch in range(epochs):
            overall_loss, overall_loss_rec, overall_loss_cls, overall_loss_kl = 0, 0, 0, 0
            n_batches = 0  # Keep track of the number of batches processed

            for x_batch, y_batch in data_loader:
                if len(x_batch.shape) > 2:
                    x_batch = x_batch.view(batch_size, x_batch.shape[-1]*x_batch.shape[-2]).to(self.device)
                else:
                    x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device) # Move batch to the appropriate device

                optimizer.zero_grad()

                # Forward pass: compute predicted outputs by passing batch to the model
                x_hat, y_hat, mean, log_var = self(x_batch)

                # Calculate the loss for the current batch
                reconstruction_loss, classification_loss, KLD = self.loss_function(x_batch, y_batch, x_hat,
                                                                                   y_hat, mean, log_var)
                loss = reconstruction_loss + classification_loss + KLD

                overall_loss += loss.item()  # Accumulate the loss
                overall_loss_rec += reconstruction_loss.item()  # Accumulate the loss
                overall_loss_cls += classification_loss.item()  # Accumulate the loss
                overall_loss_kl += KLD.item()  # Accumulate the loss

                n_batches += 1

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # Perform a single optimization step (parameter update)
                optimizer.step()

            # Print average loss for the epoch
            _, _, _, metrics_train = self.evaluate(x, y)
            _, _, _, metrics_test = self.evaluate(x_test, y_test)
            print(f"\tEpoch {epoch + 1}: \t"
                  f"Average Loss:  {overall_loss / (n_batches * batch_size)}"
                  f"\t REC Loss:  {overall_loss_rec / (n_batches * batch_size)}"
                  f"\t CLS Loss:  {overall_loss_cls / (n_batches * batch_size)}"
                  f"\t KL Loss:  {overall_loss_kl / (n_batches * batch_size)}"
                  f"\t ACC train:  {metrics_train['accuracy']}"
                  f"\t ACC test:  {metrics_test['accuracy']}")

        return history

    def predict(self, x, batch_size=500):
        self.eval()
        dataset = TensorDataset(x)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        x_hat_list, y_hat_list, mean_list, log_var_list = [], [], [], []
        for x_batch, in data_loader:
            if len(x_batch.shape) > 2:
                x_batch = x_batch.view(batch_size, x_batch.shape[-1] * x_batch.shape[-2]).to(self.device)
            else:
                x_batch = x_batch.to(self.device)
            x_hat, y_hat, mean, log_var = self(x_batch)
            x_hat_list.append(x_hat)
            y_hat_list.append(y_hat)
            mean_list.append(mean)
            log_var_list.append(log_var)

        x_hat = torch.concat(x_hat_list, axis=0).cpu().detach().numpy()
        y_hat = torch.concat(y_hat_list, axis=0).cpu().detach().numpy()
        mean = torch.concat(mean_list, axis=0).cpu().detach().numpy()
        log_var = torch.concat(log_var_list, axis=0).cpu().detach().numpy()

        return x_hat, y_hat, mean, log_var

    def evaluate(self, x_test, y_test, save_path=None):
        x_hat, y_hat, mean, log_var = self.predict(x_test)
        predicet_label = np.argmax(y_hat, axis=-1)


        # print(report)
        metrics = {
            'accuracy': accuracy_score(y_test, predicet_label),
            'precision': precision_score(y_test, predicet_label, average='weighted', zero_division=1),
            'recall': recall_score(y_test, predicet_label, average='weighted', zero_division=1),
            'f1_score': f1_score(y_test, predicet_label, average='weighted', zero_division=1)
        }
        if save_path is not None:
            report = classification_report(y_true=y_test, y_pred=predicet_label, zero_division=0)
            # Save the report to a text file
            with open(save_path + 'classification_report_autoencoder.txt', "w") as file:
                file.write(report)

        return y_hat, mean, log_var, metrics
