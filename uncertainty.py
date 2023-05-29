import seaborn as sns
import tensorflow as tf
import evidential_deep_learning as edl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel
from scipy.spatial.distance import cdist
from scipy.stats import hmean as hm
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = "cuda:0"
    print(f"GPU detected. Running on {device}")
else:
    device = "cpu"
    print("No GPU detected. Running on CPU")




class UncertaintyPredict:
    """
    :param model: type of the model for uncertainty prediction, 'DONN',
    'pDONN', 'BNN' or 'GP'.
    'DONN' is a Double Output Neural Network,
    'pDONN' is a Prior Double Output Neural Network
    'BNN' uncertainty shows the confidence of model on new objects as well,
    by training stochastic Baysian NN and measuring mean and variance of the prediction
    'GP' is a Gaussian Process Regression with RBF kernel.
    :param num_layers: number of hidden layers in the model
    :param n_hidden: number of neurons in each hidden layer
    :param noise: noise level for GP

    """

    def __init__(self, model_kind='',
                 num_layers=3, n_hidden=64, noise=0):
        self.model_kind = model_kind
        self.num_layers = num_layers
        self.n_hidden = n_hidden
        self.convert = tf.convert_to_tensor
        self.noise = noise
        self.n_train = 0
        if self.model_kind == 'DONN':
            self.model = tf.keras.Sequential(
                [tf.keras.layers.Dense(self.n_hidden,
                                       activation='relu')] * self.num_layers +
                [tf.keras.layers.Dense(2)]  # Output = (μ, ln(σ))
            )
        if self.model_kind == 'pDONN':
            self.model = tf.keras.Sequential(
                [tf.keras.layers.Dense(self.n_hidden,
                                       activation='relu')] * self.num_layers +
                [edl.layers.DenseNormalGamma(1)]  # Output = (μ, v, α, 	β)
            )

        if self.model_kind == 'BNN':
            self.convert = lambda x: torch.from_numpy(x).float().to(device)
            layers = [bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1, out_features=n_hidden),
                      nn.ReLU()] + \
                     [bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_hidden, out_features=n_hidden),
                      nn.ReLU()] * (num_layers - 1) + \
                     [bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_hidden, out_features=1)]
            self.model = nn.Sequential(*layers)

        if self.model_kind == 'GP':
            kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            if self.noise:
                kernel += WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-10, 1e+1))
            self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    def loss(self, y_true, y_pred):
        if self.model_kind == 'DONN':
            mu = y_pred[:, :1]  # first output neuron
            log_sig = y_pred[:, 1:]  # second output neuron
            sig = tf.exp(log_sig)  # undo the log

            return tf.reduce_mean(2 * log_sig + ((y_true - mu) / sig) ** 2)

        if self.model_kind == 'pDONN':
            return edl.losses.EvidentialRegression(y_true, y_pred, coeff=1e-2)

    def train(self, X, y, verbose=0, lr=0.01, num_ep=10000):
        X, y = self.convert(X.reshape(-1, 1)), self.convert(y.reshape(-1, 1))
        self.n_train = X.shape[0]
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if self.model_kind in {'DONN', 'pDONN'}:
            self.model.compile(loss=self.loss,
                               optimizer=optimizer)
            self.model.fit(X, y, epochs=num_ep, verbose=verbose)

        if self.model_kind == 'BNN':
            self.model.to(device)
            mse_loss = nn.MSELoss()
            kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
            kl_weight = 0.01

            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            for step in range(num_ep):
                pre = self.model(X)
                mse = mse_loss(pre, y)
                kl = kl_loss(self.model)
                cost = mse + kl_weight * kl

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

        if self.model_kind == 'GP':
            self.model.fit(X, y)

    def predict(self, X):
        X = self.convert(X.reshape(-1, 1))

        if self.model_kind == 'GP':
            return self.model.predict(X, return_std=True)

        pred = self.model(X)

        if self.model_kind == 'DONN':
            mu, sigma = pred[:, 0], np.exp(pred[:, 1])

        if self.model_kind == 'pDONN':
            mu, v, alpha, beta = tf.split(pred, 4, axis=-1)
            mu = mu[:, 0]
            sigma = np.sqrt(beta / (v * (alpha - 1)))
            sigma = np.minimum(sigma, 1e3)[:, 0]

        if self.model_kind == 'BNN':  # todo: optimise this step
            self.model.to(device)
            n_times = 1000
            pred = np.array([self.model(X).data.cpu().numpy() for k in range(n_times)])
            pred = pred[:, :, 0].T
            mu = pred.mean(axis=1)
            sigma = pred.std(axis=1)

        return mu, sigma

    def plot_confidence_interval(self, X_train, y_train, X_test, y_test, sgm_levels=4, name=''):
        if len(X_test) == 0:
            X_test = tf.identity(X_train)

        # Make the prediction
        y_test = y_test[np.argsort(X_test)]
        X_test = np.sort(X_test)
        y_min, y_max = y_test.min() * 1.5, y_test.max() * 1.5
        mu, sigma = self.predict(X_test)
        fig, ax = plt.subplots()
        # sns.lineplot(x=[X_train.min(), X_train.min()], y=[y_min, y_max], linestyle='dashed', c='b', alpha=0.5)
        # sns.lineplot(x=[X_train.max(), X_train.max()], y=[y_min, y_max], linestyle='dashed', c='b', alpha=0.5)
        sns.lineplot(x=X_test, y=mu, label=r'\boldmath$\mu$', alpha=0.5, c='b', ax=ax, linewidth=2.5)
        sns.lineplot(x=X_test, y=y_test, linestyle='dashed',
                     alpha=0.2, c='black', label='ground truth', ax=ax)
        ax.scatter(X_train, y_train, label='train data', alpha=0.5, c='r', s=5)
        # Calculate the confidence interval bounds
        for k in range(1, sgm_levels + 1):
            ci_low = mu - sigma * k
            ci_high = mu + sigma * k
            ci_name = rf'\boldmath$\mu \pm {k} \cdot\sigma$'
            # Plot the data and prediction with confidence interval
            #         sns.lineplot(x=x_new, y=ci_low, c='b', alpha=0.4)
            #         sns.lineplot(x=x_new, y=ci_high, c='b', alpha=0.4)
            ax.fill_between(X_test, ci_low, ci_high, alpha=0.1, color='purple')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Prediction with Confidence Interval')
        ax.legend()
        plt.ylim([y_min - 0.2, y_max + 0.2])
        plt.title(f'Confidence interval for model with {self.model_kind} uncertainty')
        # plt.savefig(f'Images/{self.model_kind}_{name}.svg', format='svg')
        # plt.draw()
        # plt.show()
        return fig

    # todo: write this function
    def get_scores(self, X_valid, y_valid):
        mu_valid, sigma_valid = self.predict(X_valid)
        return {r'MSE($y$, $\hat{\mu}$)': mse(y_valid, mu_valid),
                r'Mean($\hat{\sigma}^2$)': np.square(sigma_valid).mean(),
                r'MSE($|y - \mu|$, $\hat{\sigma}$)': mse(np.abs(y_valid - mu_valid), sigma_valid),
                r'n_train': self.n_train}


def unsupervised_std(X_train, X_test):
    # cdist is the euclidean distance between two sets of points
    d = cdist(X_test[:, None], X_train[:, None])
    # hm is the harmonic mean
    unc = hm(d, axis=1)
    # normalise the uncertainty
    unc_norm = unc / unc.std() ** 2
    return unc_norm
