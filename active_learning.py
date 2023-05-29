"f^(x) -> mu, sigma"
#
# '''
# data_generator returns X_pool, X_train, X_valid
# X_pool: a large number of uniformly distributed values on a specific range
# X_train: a small num of training samples for initial model training.
# Must not intersect with X_pool (probably)
# todo: think about validation samples
# '''
from uncertainty import *

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import argrelextrema
from matplotlib.backends.backend_pdf import PdfPages

%config InlineBackend.figure_formats = ['svg']
sns.set_theme()
%matplotlib inline


plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
from tqdm.notebook import tqdm
metrics = [r'MSE($y$, $\hat{\mu}$)',
               r'Mean($\hat{\sigma}^2$)',
               r'MSE($|y - \mu|$, $\hat{\sigma}$)',
               r'n_train']


def plot_metric(scores):
    fig, ax = plt.subplots()
    for k, y in scores.items():
        sns.lineplot(x=np.arange(len(y)), y=y, label=k, ax=ax)
    ax.legend()
    ax.set_xlabel('active learning step')
    ax.set_ylabel('log(MSE)')
    ax.set_yscale('log')
    return fig
#     plt.show()


def active_learning(data_generator, n_samples, f_hat, f_true,
                    n_steps=1000, k=4, choose_new='al',
                    save_pdf=False, pdf_name='report'):
    print('Generating data...')
    X_pool, X_train, X_valid = data_generator(n_samples=n_samples)
    print(f"Shapes of X_pool, X_train, X_valid: {X_pool.shape, X_train.shape, X_valid.shape}")

    print('Calculating the values of f_true...')
    y_train, y_valid = f_true(X_train), f_true(X_valid)  # might be expensive

    # todo: initialise model from checkpoints
    print('Initialising the model by initial training...')
    f_hat.train(X_train, y_train, num_ep=5000)

    #     metrics = ['MSE_mu', 'mean_sgm', 'MSE_sgm']
    curr_scores = f_hat.get_scores(X_valid, y_valid)
    scores = {key: [val] for key, val in curr_scores.items()}
    print("Starting active learning...")
    if save_pdf:
        pp = PdfPages(f'pdf_reports/{pdf_name}_{choose_new}.pdf')
        fig = f_hat.plot_confidence_interval(X_train=X_train, y_train=y_train, X_test=X_valid, y_test=y_valid)
        plt.title(f'Model initially trained on {len(X_train)} data points')
        pp.savefig(fig)
    for i in range(n_steps):
        X_pool.sort()
        if choose_new == 'al':
            mus, sigmas = f_hat.predict(X_pool)  # todo: replace by gradient descent
            #         X_new = np.random.choice(X_pool[argrelextrema(sigmas, np.greater)], k)
            X_new = X_pool[argrelextrema(sigmas, np.greater, mode='wrap')]
            if len(X_new) >= k:
                mus, sigmas = f_hat.predict(X_new)
                X_new = X_new[np.argsort(sigmas)[::-1]][:k]
            else:
                X_new = X_pool[np.argsort(sigmas)[::-1]][:k]
        if choose_new == 'random':
            X_new = np.random.choice(X_pool, k)

        y_new = f_true(X_new)

        X_train = np.hstack([X_train, X_new])
        y_train = np.hstack([y_train, y_new])

        f_hat.train(X_train, y_train, num_ep=1000)

        # plotting progress
        if save_pdf:
            fig = f_hat.plot_confidence_interval(X_train=X_train, y_train=y_train, X_test=X_valid, y_test=y_valid)
            plt.scatter(X_new, y_new, marker='*', alpha=0.7, c='r', label='New points')
            plt.legend()
            plt.title(f'Step {i + 1}, {len(X_new)} additional points')
            pp.savefig(fig)

        # update metrics
        for key, val in f_hat.get_scores(X_valid, y_valid).items():
            scores[key].append(val)

    if save_pdf:
        pp.savefig(plot_metric(scores))
        pp.close()
    return f_hat, scores


def data_generator(x_min=-3, x_max=6, n_samples=1000):
    #     train_size, valid_size = n_samples // 10, n_samples // 10
    train_size, valid_size = 10, 100
    X_pool = np.random.uniform(x_min, x_max, (n_samples,))
    np.random.shuffle(X_pool)
    X_train = X_pool[:train_size]
    #     X_valid = X_pool[train_size + 1 : valid_size + train_size]
    X_valid = np.random.uniform(x_min, x_max, (valid_size,))

    X_pool = X_pool[len(X_train):]
    return X_pool, X_train, X_valid


def test_err_ntrain(data_generator, f_true, model_kind, step):
    X_pool, X_train, X_valid = data_generator(n_samples=1000)
    scores = {}
    for n_train in np.arange(10, 47, 2):
        X_train = np.random.choice(X_pool, n_train)
        y_train = f_true(X_train)
        f_hat = UncertaintyPredict(model_kind=model_kind)
        f_hat.train(X_train, y_train, num_ep=1000)
        y_valid = f_true(X_valid)
        # update metrics
        for key, val in f_hat.get_scores(X_valid, y_valid).items():
            scores.get(key, default=[]).append(val)
    return scores


def func(x):
    if x < 0:
        return 0.
    else:
        return np.sin(x)


f_true = np.vectorize(func)
f_hat = UncertaintyPredict(model_kind='GP')

gp_trained, gp_res = active_learning(data_generator, 1000, f_hat, f_true, n_steps=48, k=1,
                                     pdf_name='GP_lin_sin_k=1')
gp_res1 = test_err_ntrain(data_generator, f_true, model_kind='GP')
fig , ax = plt.subplots(1, 3, sharey=True, figsize=(10, 5))

for i in range(3):
    ax[i].plot(gp_res['n_train'], gp_res[metrics[i]], label='active learning')
    ax[i].plot(gp_res1['n_train'], gp_res1[metrics[i]], label='random sampling')
    ax[i].set_xlabel('n_train')
    ax[i].set_ylabel(f"log(metrics[i])")
    ax[i].set_title(f"{metrics[i]}")
    ax[i].legend()
    ax[i].set_yscale('log')
plt.savefig('images/n_train_gp.png')
plt.show()
