import numpy.random as nr  # type: ignore
import numpy as np
from numba import njit, typed  # type: ignore

import scipy.stats as sts  # type: ignore


def get_all_keys(srs):
    return sorted({k for doc in srs for k in doc})


def typed_dict(d):
    nd = typed.Dict()
    for k, v in d.items():
        nd[k] = v
    return nd


def typed_list(l):
    tl = typed.List()
    for e in l:
        tl.append(e)
    return tl


def sample_arr(n_reps, n_per_rep, p_norm, ks, quants=[.25, .5, .75]):
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


@njit
def dict2array(d, k2ix, arr):
    for k, v in d.items():
        ix = k2ix[k]
        arr[ix] = v
    return arr


@njit
def dicts2arrays(ds, k2ix):
    D = len(ds)
    arr = np.zeros((D, len(k2ix)), np.int64)

    for i in range(D):
        dict2array(ds[i], k2ix, arr[i])
    return arr


def prob2dir_alpha(p):
    pp1 = p + 1
    sa = pp1.sum(axis=1)[:, None]
    return pp1 / sa


@njit
def gmean_hist1d_(hist_a_cts, lks):
    s, n = 0, 0
    for ct, lk in zip(hist_a_cts, lks):
        s += ct * lk
        n += ct
    return s, n


@njit
def gmean_hist1d(hist_a_cts, lks):
    s, n = gmean_hist1d_(hist_a_cts, lks)
    return np.exp(s / n)


@njit
def gmean_hist2d(hist_a_cts_arr, ks, eps=1e-6):
    lks = np.log(ks + eps)
    s, n = 0, 0
    for hist in hist_a_cts_arr:
        s_, n_ = gmean_hist1d_(hist, lks)
        s += s_
        n += n_
    return np.exp(s / n)


def enumerate_hists(counts, ks, eps=1e-6):
    """
    counts=h.p_arr
    ks=h.ks
    """
    enumerated_vals = [
        e for hist in counts for val, ct in zip(ks, hist) for e in [val] * ct
    ]

    enumerated_vals = np.array(enumerated_vals)
    return sts.gmean(enumerated_vals + eps)


class Hist:
    def __init__(self, s):
        self.name = s.name

        self.all_keys = get_all_keys(s)
        self.k2ix = k2ix = typed_dict(
            {k: i for i, k in enumerate(self.all_keys)}
        )
        self.ix2k = typed_dict({v: k for k, v in self.k2ix.items()})
        self.ks = np.asarray(sorted(k2ix))

        self.dct_lst = ds = typed_list(s)
        self.p_arr = p = dicts2arrays(ds, k2ix)
        self.p_norm = p / p.sum(axis=1)[:, None]
        self.p_norm_sq = p / (p.sum(axis=1) ** .5)[:, None]
        self.dir_alpha = prob2dir_alpha(p)
        self.dir_alpha_all = p.sum(axis=0)
        self.dir_alpha_all_sqrt = self.p_norm_sq.sum(axis=0)
        # self.dir_alpha_all = (
        #     self.dir_alpha_all / self.dir_alpha_all.sum() * 100
        # )

    def sample_geo_means1(self, n_hists):
        hists = nr.dirichlet(self.dir_alpha_all, size=n_hists)
        return gmean_hist2d(hists, self.ks, eps=1e-6)

    def sample_geo_means(self, n_hists, n_reps):
        return np.array(
            [self.sample_geo_means1(n_hists) for _ in range(n_reps)]
        )


def test_gmean_hist2d(counts, ks, eps=1e-6):
    est1 = enumerate_hists(counts, ks, eps)
    est2 = gmean_hist2d(counts, np.array(ks), eps=eps)
    assert est1 == est2
