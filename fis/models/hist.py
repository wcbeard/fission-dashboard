import itertools as it
from typing import Dict, List


import numpy.random as nr  # type: ignore
import numpy as np
from numba import njit, typed  # type: ignore
import pandas as pd
import scipy.stats as sts  # type: ignore

import fis.utils.fis_utils as fu


def identity(x):
    return x


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


@njit
def seed(n):
    nr.seed(n)


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

    @staticmethod
    def normp(p, fn=identity):
        """
        Normalize the histograms [n x K]
        - K: cardinality of support
        - n: # clients
        if `fn` is identity function, then each row
        will sum to 1. This gives every client equal
        weight. This can be transformed so a user
        w/ high counts contributes a higher weight
        """
        if fn is None:
            return p
        hist_sum = p.sum(axis=1)
        hist_sum = fn(hist_sum)
        return p / hist_sum[:, None]

    def to_dict(self, fn=identity):
        vs = self.normp(self.p_arr, fn=fn).sum(axis=0)
        return dict(zip(self.ks, vs))


def est_statistic(
    dir_dict: Dict[int, float],
    n_hists=10_000,
    client_draws=10,
    stat_fn=np.mean,
    quantiles=[.05, .5, .95],
):
    dir_dict = sorted(dir_dict.items())
    ks, alpha = zip(*dir_dict)
    # h1.dir_alpha_all_sqrt
    sampled_hists = nr.dirichlet(alpha, size=n_hists)
    stats = []
    for hist in sampled_hists:
        samps_i = nr.choice(ks, p=hist, size=client_draws)
        stats.append(stat_fn(samps_i))
    if quantiles is not None:
        qs = np.quantile(stats, quantiles)
        return dict(zip(map(fu.rn_prob_col, quantiles), qs))
    return np.array(stats)


#########################
# Multimodal Histograms #
#########################
def est_statistic_mm_beta(
    dir_dict: Dict[int, float], bins: List[int], quantiles=[.05, .5, .95]
):
    """
    return df with columns==quantiles, index == bins.
    """
    a_plus_b = sum(dir_dict.values())
    bin_ab_dct = {}
    for bin in bins:
        a = dir_dict.get(bin, 0) + 1
        b = a_plus_b - a
        bin_ab_dct[bin] = (a, b)
    a_other = sum([v for k, v in dir_dict.items() if k not in bins]) + 1
    bin_ab_dct[-1] = (a_other, a_plus_b - a_other)

    quantiles_dct = {
        bin: sts.beta(a, b).ppf(quantiles)
        for bin, (a, b) in bin_ab_dct.items()
    }
    df = pd.DataFrame(quantiles_dct, index=quantiles).T
    df.index.name = "bins"
    return df


def mm_hist_quantiles_beta(df, hcol="gc_slice_during_idle", bins=[0, 98, 100]):
    df = df[["date", "br", hcol]]
    res = pd.concat(
        [
            (
                est_statistic_mm_beta(h, bins)
                .rename(columns=fu.rn_prob_col)
                .assign(br=br, date=date)
                .reset_index(drop=0)
            )
            for date, br, h in df.itertuples(index=False)
        ],
        axis=0,
        ignore_index=True,
    )
    return res


#########
# Tests #
#########
def test_gmean_hist2d(counts, ks, eps=1e-6):
    est1 = enumerate_hists(counts, ks, eps)
    est2 = gmean_hist2d(counts, np.array(ks), eps=eps)
    assert est1 == est2


def test_normp():
    a = np.array([[2, 2], [8, 8]])
    n = Hist.normp(a, fn=identity)
    should_be = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert (n == should_be).all()
    ns = Hist.normp(a, fn=np.sqrt)
    should_be2 = np.array([[1, 1], [2, 2]])
    assert (ns == should_be2).all()
    return ns


def test_rand_choice():

    seed(0)
    ks = np.r_[0, 1, 3, 5, 7, 9, 11]
    vs = np.r_[0.09, 0.01, 0.01, 0.01, 0.01, 0.04, 0.01]
    bunch_of_samps = rand_choice(ks, 100_000, vs)
    d = pd.DataFrame(
        {
            "emp": pd.Series(bunch_of_samps)
            .value_counts(normalize=0)
            .sort_index()
            .pipe(lambda x: x.div(x.sum())),
            "base": vs,
        }
    ).assign(dif=lambda df: (df.emp - df.base).abs())
    print(f"Max diff: {d.dif.max():.1%}")
    return d
