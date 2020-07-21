import numpy.random as nr
import numpy as np


def sample_arr(n_reps, n_per_rep, p_norm, ks, quants=[.05, .5, .95]):
    # res = np.empty((n_reps, n_per_rep))
    ixs = np.arange(len(p_norm))
    hist_ixs = nr.choice(ixs, size=n_reps, replace=True)
    hists = p_norm[hist_ixs]

    all_samps = []
    for hist in hists:
        all_samps.extend(nr.choice(ks, size=40, p=hist))
    return np.quantile(all_samps, quants)


def sample_arr2(n_reps, n_per_rep, p_norm, ks, quants=[.05, .5, .95]):
    # res = np.empty((n_reps, n_per_rep))
    ixs = np.arange(len(p_norm))
    hist_ixs = nr.choice(ixs, size=n_reps, replace=True)
    hists = p_norm[hist_ixs]

    all_samps = []
    for hist in hists:
        all_samps.extend(nr.choice(ks, size=40, p=hist))
    return np.quantile(all_samps, quants)
