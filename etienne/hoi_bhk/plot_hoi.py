import numpy as np


def plot_landscape(df, nbins=10, symetric=True, cmap='RdBu_r'):
    import matplotlib.pyplot as plt

    # get the sizes of the multiplets
    sizes = np.array(list(df['Sizes']))
    usizes = np.unique(sizes)

    # get the oinfo
    oinfo = np.array(list(df['Oinfo'] )) * np.array(list(df['Sign']))
    omin, omax = oinfo.min(), oinfo.max()
    if symetric:
        ominmax = max(abs(omin), abs(omax))
        omin, omax = -ominmax, ominmax
    print(omin, omax)
    bins = np.linspace(omin, omax, nbins + 1, endpoint=True)

    obins = np.full((nbins, len(usizes)), np.nan)
    for n_m, m in enumerate(usizes):
        oinfo_s = oinfo[sizes == m]
        for n_b in range(nbins - 1):
            indices = np.logical_and(
                bins[n_b] <= oinfo_s, oinfo_s <= bins[n_b + 1])
            if np.any(indices):
                # obins[n_b, n_m] = oinfo_s[indices].mean()
                obins[n_b, n_m] = indices.sum()
            # else:
            #     obins[n_b, n_m] = 0.

    plt.pcolormesh(usizes, bins, obins, cmap=cmap)  # vmin=omin, vmax=omax
    plt.xlabel('Multiplet size')
    plt.ylabel('Oinfo')
    plt.colorbar()

    return plt.gca()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    file = 'DA_21_genes.xlsx'
    path = '/run/media/etienne/DATA/Toolbox/BraiNets/hoi_bhk/etienne/hoi_bhk/'

    df = pd.read_excel(path + file)

    obins = plot_landscape(df, nbins=60, cmap='turbo', symetric=False)


    plt.show()
