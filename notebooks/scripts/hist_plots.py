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
    "..", "/Users/wbeard/repos/fis/fis/", "/Users/wbeard/repos/fis/", "~/repos/myutils/",
)
add_path("/Users/wbeard/repos/dscontrib-moz/src/")


# %%
# HIDDEN
def jlab_kernel():
    import sys
    return any(c.startswith('/Users/wbeard/Library/Jupyter') for c in sys.argv)
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
from collections import OrderedDict
from functools import partial, lru_cache, wraps

import altair as A
import pandas as pd
import scipy.stats as sts

from fis.models import hist as hu
import fis.data.load_agg_hists as loh
from fis.utils import bq
import fis.utils.vis as vz
import fis.utils.fis_utils as fu

# %%
dfh2_ = bq.bq_query(loh.dl_agg_query())
dfh2 = loh.proc_hist_dl(dfh2_)

# %%
# HIDDEN
if jlab_kernel():
    s = dfh2.unq_sites_per_doc
    h = s[0]
    ss = pd.Series(s[0]).sort_index()
    samps = hu.est_statistic(ss, stat_fn=gmean, quantiles=None)

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
hist_dfs = OrderedDict(
    [(hcol, summarize_hist_df(dfh2, dfh2[hcol])) for hcol in loh.hist_cols]
)

# %%
import seaborn as sns

sns.kdeplot(samps)

# %% [markdown]
# # How to deal with multimodal

# %%
from numba import typed

mc = 'gc_slice_during_idle'

# %%
loh.multimodal_histograms

# %%
dfh2[:3]


# %%
# for mc in loh.multimodal_histograms:
#     sr = dfh2[mc]
#     for hist, date, br in dfh2[[mc, 'date', 'br']].itertuples(index=False):
#         break
# #         sr = dfh2[mc]
# #         s = sr[0]
#         break
    
# # samps = hu.est_statistic(hist, stat_fn=gmean, quantiles=None)

# %%
def topn_hist(h, n=3):
    s = (
        pd.Series(h)
        .sort_values(ascending=False)
        .pipe(lambda x: x.div(x.sum()))
        .mul(100)
        .cumsum()
        .round(1)
    )
    return s[:n]


h1 = sr[0]
topn_hist(h1)

# %%
from typing import Dict



# %%
h = dfh2[mc][0]
hs = Series(h).sort_index()
ks = np.array(sorted(h))
# samps_i = est_statistic(h)

# da = hu.DctArr([0, 99, 100])
# da.ix_lookup

# %%
bss_qs = hu.est_statistic_mm(h, [0, 98, 100], client_draws=1_000, n_hists=13)

# bss_qs.index = bss_qs.index.rename('bins')
bss_qs.reset_index(drop=0)
# .T

# %%
fu.s(h)

# %%
qs_summ = hu.mm_hist_quantiles(
    df=dfh2[:4],
    hcol="gc_slice_during_idle",
    bins=[0, 100],
    n_hists=50,
    client_draws=100,
)
qs_summ

# %%
qs_summ2 = hu.mm_hist_quantiles2(
    df=dfh2,
    hcol="gc_slice_during_idle",
    bins=[0, 100],
)
qs_summ2[:3]

# %%
b100 = qs_summ2.query("bins == 100")

# %%
# qs_summ2.br.drop_duplicates()

# %%
_h = dfh2.query("br == 'enabled' & date == '2020-06-12'")[mc].iloc[0]
# _h

# %%
_q = hu.est_statistic_mm_beta(_h, [0, 100])
_q

# %%
loh.multimodal_histograms

# %%
vz.plot_errb(b100, ytitle="% == 100", zero=False)

# %%
# bss, = hu.est_statistic_mm(h, [0, 98, 100], client_draws=11)

# %%
# hs

# %% jupyter={"outputs_hidden": true}
bss

# %%
bss.quantile([.05, .5, .95])

# %%
samps_i

# %%
from numba.typed import typeddict
from numba import int64, float64

@njit
def _summarize_multimodal_hist(samps, bin_cts):
    "With dict"
    for s in samps:
        if s in bin_cts:
            bin_cts[s] += 1
        else:
            bin_cts[-1] += 1
    return bin_cts

@njit
def _normalize_mm_hist_summary(dct):
    dnorm = typeddict.Dict()
    s = 0
    for v in dct.values():
        s += v
    for k, v in dct.items():
        dnorm[k] = v / s
    return dnorm

bin_cts = typeddict.Dict()
bin_cts.update({0: 0, 100: 0, -1: 0})

# %%




# %%

# %%
ks, vs = map(np.array, zip(*sorted(h.items())))
vs = vs / vs.sum()
# ks, vs

# %%
ks


# %%
@njit
def get2(self, k, default):
    v = self.get(k)
    if v is None:
        return default
    return v

typeddict.Dict.get2 = get2

# %%
da.ix_lookup.get(-3, 1)

# %%


bin_cts = typeddict.Dict(dcttype=typeddict.DictType(int64, float64))
bin_cts.update({0: 0, 100: 0, -1: 0})

summarize_multimodal_hist_arr(samps_i, da.ix_lookup)

# %%
import numba as nb


# %%
@njit
def est_statistic_mm(dir_alphas, ix_lookup, ks, client_draws=10, norm=True):
    n = len(dir_alphas)
    res = np.empty((n, len(ix_lookup) + 1))
    for i in range(n):
        client_samples = hu.rand_choice(ks, size=client_draws, prob=dir_alphas[i])
        #         return client_samples
        res[i] = hu.summarize_multimodal_hist_arr(
            client_samples, ix_lookup, norm=norm
        )
    return res

binned_samps = hu.draw_mm_samps_from_dir(_alpha_dir, da.ix_lookup, ks,)


# %%
def est_statistic_mm(
    dir_dict: Dict[int, float],
    n_hists=10_000,
    client_draws=10,
    stat_fn=np.mean,
    quantiles=[.05, .5, .95],
):
    #dir_dict = sorted(dir_dict.items())
    ks, alpha = map(np.array, zip(*sorted(dir_dict.items())))
    
#     ks, alpha = zip(*dir_dict)
#     print(alpha)
    # h1.dir_alpha_all_sqrt
    sampled_hists = nr.dirichlet(alpha, size=n_hists)
    binned_samps = hu.draw_mm_samps_from_dir(sampled_hists, da.ix_lookup, ks,)
#     binned_samps = DataFrame(binned_samps, columns=da.bins)
#     if quantiles is not 1:
#         qs = np.quantile(stats, quantiles)
#         return dict(zip(map(fu.rn_prob_col, quantiles), qs))
    
    return binned_samps
    stats = []
    for hist in sampled_hists:
        samps_i = nr.choice(ks, p=hist, size=client_draws)
        return samps_i
        stats.append(stat_fn(samps_i))
    if quantiles is not None:
        qs = np.quantile(stats, quantiles)
        return dict(zip(map(fu.rn_prob_col, quantiles), qs))
    return np.array(stats)

# _alpha_dir = 
bss = est_statistic_mm(h)

# %%
da.bins

# %%
bss

# %%
_alpha_dir

# %%
da._ix_lookup

# %%
da.ix_lookup

# %%
DataFrame(binned_samps, )

# %%
np.quantile(binned_samps, [.1, .5, .9], axis=0)

# %%
samps_i

# %%
da.ix_lookup

# %%
da.ix_lookup.get(9, -1)

# %%
bin_cts._dict_type.key_type

# %% jupyter={"outputs_hidden": true}
summarize_multimodal_hist(samps_i, bin_cts)

# %%
normalize_mm_hist_summary(bin_cts)

# %%

# %%

# %%

# 

# %%
# dct = typeddict.Dict()
dct.setdefault('a', 1)

# %%
dct

# %%
typeddict.Dict()
# typeddict.typeddict_call?

# %%
import dscontrib as dsc

# %%
plt
sns
mu.import_name('plt')


# %%
def multimodal_hist_samples(hist_df, mm_hist_col):
    dfs = []
    print('-' * len(hist_df) + '|')
    for hist, date, br in hist_df[[mc, 'date', 'br']].itertuples(index=False):
        print('.', end='')
        samps = hu.est_statistic(hist, stat_fn=gmean, quantiles=None)
        df = DataFrame(dict(samps=samps,)).assign(br=br, date=date)
        dfs.append(df)
#         if len(dfs) > 2:
#             break
    return pd.concat(dfs, axis=0, ignore_index=True)

dfs = multimodal_hist_samples(dfh2, mc).assign(date=fu.to_date_col('date'))

# %%

# %%
plt.rcParams.update({"font.size": 22})
plt.figure(figsize=(16, 6))
ax = plt.gca()
sns.violinplot(
    x="date",
    y="samps",
    data=dfs,
    hue="br",
    split=True,
    order=sorted(dfs.date.unique()),
    inner="quart",
)
sns.despine(left=True)
plt.xticks(rotation=75)
plt.legend(loc="lower left")

# %% jupyter={"outputs_hidden": true}
Series(hist).sort_index()

# %%
dfs[:3]

# %%
d2s = pd.concat(
    [
        DataFrame(
            {"samps": hist.est_statistic(h, stat_fn=gmean, quantiles=None)}
        ).assign(date=date)
        for date, h in dfh2[["date", mc]][:2].itertuples(index=False)
    ],
    axis=0,
    ignore_index=True,
)

# %%
d2s = d2s.assign(date=lambda df: pd.to_datetime(df.date).dt.date)

# %%

# %%
sns.violinplot(x='date', y='samps', data=d2s)

# %%
Series(hist).sort_index().pipe(lambda df: df / df.sum()).mul(100).round(0).astype(int).pipe(lambda x: x[x > 0])

# %%
loh.hist_cols

# %%
loh.multimodal_histograms

# %%
d2s[:3]

# %%
from vega_datasets import data

data.cars()[:3]


# %%
def _pl(pdf, x="Origin", y="Miles_per_Gallon"):
    import altair as alt
    from vega_datasets import data

    h = (
        alt.Chart(pdf)
        .transform_density(y, as_=[y, "density"],
#                            extent=[5, 50],
                           groupby=[x],)
        .mark_area(orient="horizontal")
        .encode(
            y=f"{y}:Q",
            color=f"{x}:N",
            x=alt.X(
                "density:Q",
                stack="center",
                impute=None,
                title=None,
                axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
            ),
            column=alt.Column(
                f"{x}:T",
                header=alt.Header(
                    titleOrient="bottom", labelOrient="bottom", labelPadding=0
                ),
            ),
        )
        .properties(width=100)
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
        .interactive()
    )
    return h


# _pl(data.cars())
_pl(d2s, x="date", y="samps")

# %%
p.data.date[0]

# %%

# %%
_pl(d2s, x="date", y="samps")

# %%
d2s = {
    date: hist.est_statistic(h, stat_fn=gmean, quantiles=None)
    for date, h in dfh2[['date', mc]][:2].itertuples(index=False)
}

# %% jupyter={"outputs_hidden": true}
DataFrame(d2s).stack().reset_index(drop=0)

# %%
for tup in dfh2[['date', mc]][:2].itertuples(index=False):
    break

tup

# %%
a, b = tup


# %%
samps

# %%
# HIDE CODE

A.data_transformers.enable('default')

plots = [
    vz.plot_errb(hdf).properties(height=200, width=800, title=h)
    for h, hdf in hist_dfs.items()
]

A.vconcat(*plots)

# %%
# HIDE CODE
# A.data_transformers.enable('json')
A.data_transformers.enable('default')
vz.plot_errb(hdf).properties(height=200, width=800, title=s.name)

# %%

# %% [markdown]
# # Junk
# <!-- # HIDDEN -->

# %%
# HIDDEN
turtle = lambda: defaultdict(turtle)

def fn_tuple(full_fn):
    _loc, fn = os.path.split(full_fn)
    locs = _loc.split('/')[2:]
    return locs, fn


def build_dir_dicts(dirs):
    dirs = [
        d.split('/')[2:-1]
        for d in dirs
    ]
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
