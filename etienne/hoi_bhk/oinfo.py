import numpy as np
import itertools

from joblib import Parallel, delayed

from hoi_bhk.gcmi import copnorm, ent_g
from functools import partial

ent_g = partial(ent_g, biascorrect=False)


def _o_info(x, comb):
    # (n_variables, n_samples)
    nvars, _ = x.shape

    # (n - 2) * H(X^n)
    o = (nvars - 2) * ent_g(x)

    for j in range(nvars):
        # sum_{j=1...n}( H(X_{j}) - H(X_{-j}^n) )
        o += ent_g(x[[j], :]) - ent_g(np.delete(x, j, axis=0))
    comb = np.array(comb) + 1
    return o, str(comb)


def exhaustive_loop_zerolag(ts, maxsize=5, n_best=10, n_jobs=-1,
                            as_pandas=True):
    """Simple implementation of the Oinfo.

    Parameters
    ----------
    ts : array_like
        Time-series of shape (n_variables, n_samples) (e.g (n_roi, n_trials))
    """
    x = copnorm(ts)
    nvars, nsamp = x.shape

    # get the maximum size of the multiplets investigated
    if not isinstance(maxsize, int):
        maxsize = nvars
    maxsize = max(3, maxsize)

    # get the combination object
    oinfo, combinations = [], []
    for _max in range(3, maxsize + 1):
        # get all of the combinations
        all_comb = itertools.combinations(range(0, nvars), _max)

        # compute oinfo
        outs = Parallel(n_jobs=n_jobs)(delayed(_o_info)(
            x[comb, :], comb) for comb in all_comb)

        _oinfo, _combinations = zip(*outs)
        oinfo.append(_oinfo)
        combinations.append(_combinations)

    if as_pandas:
        import pandas as pd

        df = pd.DataFrame({
            'Combination': np.concatenate(combinations),
            'Oinfo': np.concatenate(oinfo),
        })
        df['Sign'] = np.sign(df['Oinfo'])
        df['Oinfo'] = np.abs(df['Oinfo'])

        df.sort_values('Oinfo', inplace=True, ascending=False)

        return df

if __name__ == '__main__':
    from scipy.io import loadmat

    ###########################################################################
    path = '/run/media/etienne/DATA/Toolbox/BraiNets/hoi_bhk/data/%s'
    file = 'Ex1_syn'
    ###########################################################################

    mat = loadmat(path % ('%s.mat' % file))
    ts = mat['data']#[:, 0:5]
    otot = mat['Otot']
    # print(otot)
    print(ts.shape)
    # exit()

    # ts = np.random.uniform(-1., 1., (10, 100))
    df = exhaustive_loop_zerolag(ts.T, maxsize=4, n_jobs=-1)
    df.to_excel('%s.xlsx' % file)
