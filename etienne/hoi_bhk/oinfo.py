import numpy as np
import pandas as pd
import itertools
from sklearn.utils import resample

from joblib import Parallel, delayed

from hoi_bhk.gcmi import copnorm, ent_g
from hoi_bhk.wrapper import jit
from hoi_bhk.stats import fdr_correction
from functools import partial

ent_g = partial(ent_g, biascorrect=False)


# @jit("f4(f4[:, :])")
def nb_ent_g(x):
    """Numba implementation of the entropy of a Gaussian variable in bits.
    """
    nvarx, ntrl = x.shape

    # covariance
    c = np.dot(x, x.T) / float(ntrl - 1)
    chc = np.linalg.cholesky(c)

    # entropy in nats
    hx = np.sum(np.log(np.diag(chc))) + 0.5 * nvarx * (
        np.log(2 * np.pi) + 1.0)
    return hx


def _o_info(x, comb, return_comb=True):
    # (n_variables, n_samples)
    # if len(comb) == 1:  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #     return nb_ent_g(x), comb

    nvars, _ = x.shape

    # (n - 2) * H(X^n)
    o = (nvars - 2) * nb_ent_g(x)

    for j in range(nvars):
        # sum_{j=1...n}( H(X_{j}) - H(X_{-j}^n) )
        o += nb_ent_g(x[[j], :]) - nb_ent_g(np.delete(x, j, axis=0))

    if return_comb:
        return o, comb
    else:
        return o


def bootci(o, x, comb, alpha, n_boots, rnd=0, n_jobs=-1):
    # bootstrap computations
    _, nsamps = x.shape
    oboot = Parallel(n_jobs=n_jobs)(delayed(_o_info)(
            resample(x.T, n_samples=nsamps, random_state=rnd + i).T, comb,
            return_comb=False) for i in range(n_boots))
    oboot = np.asarray(oboot)

    # confidence interval
    lower = np.percentile(oboot, (alpha / 2.) * 100.)
    upper = np.percentile(oboot, (1 - (alpha / 2.)) * 100.)

    # p-value inference
    indices = oboot < 0 if o > 0 else oboot > 0
    pv = (1 + np.sum(indices)) / (n_boots + 1)

    return pv, lower, upper


def exhaustive_loop_zerolag(ts, maxsize=5, n_best=10, n_jobs=-1, n_boots=None,
                            alpha=0.05):
    """Simple implementation of the Oinfo.

    Parameters
    ----------
    ts : array_like
        Time-series of shape (n_variables, n_samples) (e.g (n_roi, n_trials))
    """
    # copnorm and demean the data
    x = copnorm(ts)
    x = (x - x.mean(axis=1)[:, np.newaxis]).astype(np.float32)
    nvars, nsamp = x.shape

    # get the maximum size of the multiplets investigated
    if not isinstance(maxsize, int):
        maxsize = nvars
    maxsize = max(1, maxsize)

    # get the combination object
    oinfo, combinations, sizes = [], [], []
    for _max in range(3, maxsize + 1):
        print(f"Multiplets of size {_max}")

        # get all of the combinations
        all_comb = itertools.combinations(range(0, nvars), _max)

        # compute oinfo
        outs = Parallel(n_jobs=n_jobs)(delayed(_o_info)(
            x[comb, :], comb) for comb in all_comb)

        _oinfo, _combinations = zip(*outs)
        oinfo.append(_oinfo)
        combinations += np.array(_combinations).tolist()
        sizes += [_max] * len(_oinfo)

    # dataframe conversion
    df = pd.DataFrame({
        'Combination': combinations,
        'Oinfo': np.concatenate(oinfo),
        'Size': sizes
    })
    df.sort_values('Oinfo', inplace=True, ascending=False)

    # n_best selection
    if isinstance(n_best, int):
        # redundancy selection
        red_ind = np.zeros((len(df),), dtype=bool)
        red_ind[0:n_best] = True
        red_ind = np.logical_and(red_ind, df['Oinfo'] > 0)
        # synergy selection
        syn_ind = np.zeros((len(df),), dtype=bool)
        syn_ind[-n_best::] = True
        syn_ind = np.logical_and(syn_ind, df['Oinfo'] < 0)
        # merge both
        redsyn_ind = np.logical_or(red_ind, syn_ind)

        df = df.loc[redsyn_ind]

    # statistics
    if isinstance(n_boots, int):
        pv, cilow, cihigh = [], [], []
        for o, comb in zip(df['Oinfo'], df['Combination']):
            _pv, _low, _high = bootci(o, x[comb, :], comb, alpha, n_boots,
                                      n_jobs=n_jobs)
            pv.append(_pv)
            cilow.append(_low)
            cihigh.append(_high)
        df['pvalues'] = pv
        df['pvalues FDR'] = fdr_correction(pv)[1]
        df['ci_low'] = cilow
        df['ci_high'] = cihigh

        # get if it's significant or not
        df['Significant'] = np.sign(cilow) == np.sign(cihigh)

    return df

if __name__ == '__main__':
    from scipy.io import loadmat
    import pandas as pd

    ###########################################################################
    # file = 'Ex1_syn'
    file = 'Ex1_red'
    ###########################################################################

    if file in ['Ex1_syn', 'Ex1_red']:
        path = '/run/media/etienne/DATA/Toolbox/BraiNets/hoi_bhk/data/%s'
        mat = loadmat(path % ('%s.mat' % file))
        ts = mat['data'][:, :].T
    elif file == 'DA_21_genes':
        path = '/home/etienne/%s'
        ts = pd.read_csv(path % ('%s.csv' % file)).values.T

    # print(otot)
    print(ts.shape)

    # ts += np.random.rand(*ts.shape)
    # exit()

    # ts = np.random.uniform(-1., 1., (15, 100))
    df = exhaustive_loop_zerolag(ts, maxsize=3, n_jobs=-1, n_boots=100, n_best=20)  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print(df)
    df.to_excel('%s.xlsx' % file)
    print(df)
