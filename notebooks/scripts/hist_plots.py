# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Histograms

# %%
# HIDDEN
from boot_utes import add_path, path, reload, run_magics

add_path(
    "..",
    "/Users/wbeard/repos/fis/fis/",
    "/Users/wbeard/repos/fis/",
    "~/repos/myutils/",
)
add_path("/Users/wbeard/repos/dscontrib-moz/src/")

# %%
# HIDDEN
from collections import OrderedDict
from functools import lru_cache, partial, wraps

import altair as A
import pandas as pd
import scipy.stats as sts

import fis.data.load_agg_hists as loh
import fis.utils.fis_utils as fu
import fis.utils.vis as vz
from fis.models import hist as hu
from fis.utils import bq

H = 200
W = 800

# %%
# HIDDEN
dfh2_ = bq.bq_query(loh.dl_agg_query())
dfh2 = loh.proc_hist_dl(dfh2_)


# %%
# HIDDEN
def jlab_kernel():
    import sys

    return any(c.startswith("/Users/wbeard/Library/Jupyter") for c in sys.argv)


if jlab_kernel():
    from matplotlib import MatplotlibDeprecationWarning

    import dscontrib.wbeard as dwb
    from utils.fis_imps import *

    exec(pu.DFCols_str)
    exec(pu.qexpr_str)
    run_magics()
    # import utils.en_utils as eu; import data.load_data as ld; exec(eu.sort_dfs_str)

    mu.set_import_name(mu)
    sns.set_style("whitegrid")
    S = Series
    D = DataFrame

    import dscontrib.wbeard.altair_utils as aau

    aau.set_ds(A)

    DataFrame.pat = aau.pat


# %%
# HIDDEN
if jlab_kernel():
    %load_ext autoreload
    %autoreload 2

# %%
# HIDDEN
if jlab_kernel():
    from numba import typed

    s = dfh2.unq_sites_per_doc
    h = s[0]
    ss = pd.Series(s[0]).sort_index()
    # samps = hu.est_statistic(ss, stat_fn=gmean, quantiles=None)

    mc = "gc_slice_during_idle"

# %%
# HIDDEN
TEST = 1


def cache_dict(f):
    @lru_cache()
    def tup_f(tup):
        d = dict(tup)
        return f(d)

    @wraps(f)
    def dict_f(dict_arg):
        tuple_arg = tuple(sorted(dict_arg.items()))
        #         print(tuple_arg)
        return tup_f(tuple_arg)

    return dict_f


gmean = lambda x: sts.gmean(x + 1e-6)

agg_gmean = partial(
    hu.est_statistic,
    n_hists=100 if TEST else 10_000,
    client_draws=10,
    stat_fn=gmean,
    quantiles=[0.05, 0.5, 0.95],
)
agg_gmean_cache = cache_dict(agg_gmean)


# %%
# HIDDEN
def summarize_hist_df(df, hist_srs):
    ps = [agg_gmean_cache(h) for h in hist_srs]
    df = pd.concat([df[["date", "br", "n_cid"]], pd.DataFrame(ps)], axis=1)
    return df


# hdf = summarize_hist_df(dfh2, s)

# %%
# HIDDEN
hist_dfs = OrderedDict(
    [(hcol, summarize_hist_df(dfh2, dfh2[hcol])) for hcol in loh.hist_cols]
)

# %% [markdown]
# ## Histograms

# %%
# NO CODE
A.data_transformers.enable("default")

plots = [
    vz.plot_errb(hdf).properties(height=H, width=W, title=h)
    for h, hdf in hist_dfs.items()
]

A.vconcat(*plots)

# %% [markdown]
# # Multimodal histograms

# %%
# HIDDEN
# fu.s(dfh2['cycle_collector_slice_during_idle'].iloc[0], thresh=.05)

# %%
# HIDDEN
mean_mm_cts = pd.concat(
    [
        pd.DataFrame.from_records(dfh2[mc])
        .mean()
        .sort_index()
        .reset_index(drop=0)
        .rename(columns={"index": "k", 0: "count"})
        .assign(h=mc)
        for mc in loh.multimodal_histograms
    ],
    axis=0,
    ignore_index=True,
)


# %%
# NO CODE
def _pl(pdf):
    x = "k"
    y = "count"

    h = (
        A.Chart(pdf)
        .mark_point()
        .encode(
            x=A.X(x, title=x),
            y=A.Y(y, title=y, scale=A.Scale(zero=False)),
            tooltip=[x, y,],
        )
    ).properties(height=H, width=W / 2)

    return (h + h.mark_line()).interactive().facet(column="h", columns=3)


_pl(mean_mm_cts)

# %% [markdown]
# ## cycle_collector_slice_during_idle

# %%
# NO CODE
# bins: 0=> ~2%, 100=> ~95%
mm_est1 = hu.mm_hist_quantiles_beta(
    df=dfh2, hcol="cycle_collector_slice_during_idle", bins=[100],
)

vz.stack_bin_plots(mm_est1, h=H, w=W)

# %% [markdown]
# ## gc_slice_during_idle

# %%
# NO CODE
gcsdi = hu.mm_hist_quantiles_beta(df=dfh2, hcol="gc_slice_during_idle", bins=[0, 100],)

vz.stack_bin_plots(gcsdi, h=H, w=W)

# %% [markdown]
# # Junk
# <!-- # HIDDEN -->

# %%
# HIDDEN
pd.Series(dfh2["unq_sites_per_doc"][0]).sort_index().plot()

# %%
# HIDDEN
pd.Series(dfh2[loh.multimodal_histograms[1]][0]).sort_index().plot()

# %%
# HIDDEN
turtle = lambda: defaultdict(turtle)


def fn_tuple(full_fn):
    _loc, fn = os.path.split(full_fn)
    locs = _loc.split("/")[2:]
    return locs, fn


def build_dir_dicts(dirs):
    dirs = [d.split("/")[2:-1] for d in dirs]
    base = fu.AttrDict()
    for bc in base:
        if not bc:
            continue

    print(dirs)


# dirs = glob('../fis/**/', recursive=True)
# build_dir_dicts(dirs)


# fns = sorted([
#     fn_tuple(full_fn) + (full_fn,)
#     for full_fn in glob('../fis/**', recursive=True)
# ], key=lambda x: -len(x[0]))

# fns

# %%
