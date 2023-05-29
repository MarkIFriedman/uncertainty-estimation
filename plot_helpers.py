import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import tensorflow as tf
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

# %config InlineBackend.figure_formats = ['svg']
# sns.set_theme()
# %matplotlib inline


plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


from uncertainty import *

def df2pdf(df):
    df = df.round(4)
    fig = plt.figure()
    ax=plt.subplot(111)
    ax.axis('off')
    c = df.shape[1]
    ax.table(
             cellText=df.values,
             rowLabels=df.index,
             colLabels=df.columns,
             rowColours=['lightgray']*len(df),
             colColours=['lightgray']*len(df.columns),
             loc='center',
             bbox=[0,0,1,1],
            )
    return fig


def generate_data(func, n_samples=10, noise=0, dynamic=False, x_min=-7, x_max=7, alpha=0, gap=False):
    """Generate data from func on [x_min, x_max] with noise.
    If dynamic is True, then noise is multiplied by (1 + |x|).
    If gap is True, then there is a gap in the middle of the interval.
    """
    length = x_max - x_min
    if gap:
        n1 = int((abs(-1 - x_min) / length) * n_samples)
        n2 = n_samples - n1
        part1 = np.random.uniform(x_min, -1, (n1,))
        part2 = np.random.uniform(1, x_max, (n2,))
        X_train = np.hstack([part1, part2])
    else:
        X_train = np.random.uniform(x_min, x_max, (n_samples,))
    n_samples_test = 10 * length
    X_test = np.hstack([X_train, np.random.uniform(x_min - alpha * length,
                                                   x_max + alpha * length,
                                                   (n_samples_test,))])
    X_test.sort()
    y_test = func(X_test)
    if dynamic:
        noise = noise * (1 + np.abs(X_train))
    y_train = func(X_train) + np.random.normal(0, noise, (len(X_train),))
    return X_train, y_train, X_test, y_test


def plot_all_sigmas(X_train, X_test, sigmas, sigmas_true):
    """Plot all sigmas for all models and compare them with true sigmas.
    sigmas_true is a dict of true sigmas for all models."""
    n = len(sigmas_true) + 1
    fig, ax = plt.subplots(n, 1, sharex=True, figsize=(7, 5 * n))

    for k in sigmas_true:
        sns.lineplot(x=X_test, y=1. + sigmas[k], label=k, ax=ax[0])
    ax[0].legend()
    #     ax[0].set_ylim(-0.1, 2)
    ax[0].set_ylabel(r'$\log(1 + \sigma)$')
    ax[0].set_xlabel('x')
    ax[0].scatter(X_train, np.ones_like(X_train), c='r', marker='*', label='train data points')
    ax[0].set_title('All uncertainties')
    ax[0].set_yscale('log')

    for i, k in enumerate(sigmas_true):
        sns.lineplot(x=X_test, y=1. + sigmas[k], ax=ax[i + 1], label=r'$\hat{\sigma}$')
        sns.lineplot(x=X_test, y=1. + sigmas_true[k], ax=ax[i + 1], label=r'$y - \hat{\mu}$')
        ax[i + 1].scatter(X_train, np.ones_like(X_train), c='r', marker='*', label='train data points')
        ax[i + 1].legend()
        ax[i + 1].set_title(f'Sigma comparison for {k} model')
        ax[i + 1].set_ylabel(r'$\log(1 + \sigma)$')
        ax[i + 1].set_xlabel('x')
        ax[i + 1].set_yscale('log')

    return fig


def test_approaches(model_names=['DONN', 'pDONN', 'BNN', 'GP'],
                    model_params={}, data_genrator_params={},
                    pdf_name='untitled'):
    """Test all approaches on the same data and plot results.
    model_names is a list of model names to test.
    model_params is a dict of model parameters.
    data_genrator_params is a dict of data generator parameters."""

    params = {'num_layers': 3,
              'n_hidden': 64,
              'lr': 0.01,
              'num_ep': 10000}
    params.update(model_params)
    datagen_params = {'func': np.sin,
                        'n_samples': 10,
                        'x_min': -7,
                        'x_max': 7,
                        'noise': 0,
                        'dynamic': False,
                        'alpha': 0}
    datagen_params.update(data_genrator_params)
    x_min, x_max, n_samples = datagen_params['x_min'], datagen_params['x_max'], datagen_params['n_samples']
    pp = PdfPages(f'pdf_reports/{pdf_name} on '
                  f'[{x_min}, {x_max}] trained on {n_samples} samples.pdf')
    X_train, y_train, X_test, y_test = generate_data(**datagen_params)
    models = {}
    mus, sigmas = {}, {}
    for name in model_names:
        models[name] = UncertaintyPredict(model_kind=name,
                                          num_layers=params['num_layers'],
                                          n_hidden=params['n_hidden'],
                                          noise=datagen_params['noise'])
        models[name].train(X_train, y_train, lr=params['lr'], num_ep=params['num_ep'])
        mus[name], sigmas[name] = models[name].predict(X_test)
        plot = models[name].plot_confidence_interval(X_train, y_train, X_test, y_test)
        pp.savefig(plot)

    metrics = [r'MSE($y$, $\hat{\mu}$)',
               r'Mean($\hat{\sigma}^2$)',
               r'MSE($|y - \mu|$, $\hat{\sigma}$)',
               r'Corr($\sigma_{unsup}$, $\hat{\sigma}$)']

    res = pd.DataFrame(columns=model_names,
                       index=metrics)

    sigmas['unsupervised'] = unsupervised_std(X_train, X_test)
    sigmas_true = {}
    for m in model_names:
        sigmas_true[m] = np.abs(y_test - mus[m])
        res[m] = [mse(y_test, mus[m]),
                  np.square(sigmas[m]).mean(),
                  mse(sigmas_true[m], sigmas[m]),
                  np.corrcoef(sigmas['unsupervised'], sigmas[m])[0][1]
                  ]
    pp.savefig(plot_all_sigmas(X_train, X_test, sigmas, sigmas_true))
    pp.savefig(df2pdf(res), bbox_inches='tight')

    pp.close()