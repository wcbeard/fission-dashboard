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
# This notebook ("") tries to download all recent experiments via dbx

# %%
from boot_utes import add_path, path, reload, run_magics
add_path(
    "..", "../fis/", "~/repos/myutils/",
)
add_path("/Users/wbeard/repos/dscontrib-moz/src/")

# %%
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
# %%
from numba import typed

_hist_cols = [
    "unq_tabs",
    "unq_sites_per_doc",
    "cycle_collector_slice_during_idle",
    "gc_slice_during_idle",
    "cycle_collector",
    "cycle_collector_max_pause",
    "gc_max_pause_ms_2",
    "gc_ms",
]

# h_kw = {h: lambda df, h=h: df[h].map(arr_of_str2dict) for h in hist_cols}
# df = df_.assign(**h_kw)

# %%
import utils.fis_utils as fu
import utils.vis as vz

import fis.data.load_crap as lc
import dscontrib.wbeard.altair_utils as aau

aau.set_ds(A)

DataFrame.pat = aau.pat

# %%
df = lc.load(0, bq_read=bq_read, sqlfn="../fis/data/hist_data_proto.sql", dest="/tmp/hists.pq")
len(df)

# %%
df[:3]

# %% [markdown]
# # ETL

# %%
import fis.data.load_agg_hists as loh
from fis.utils import bq

# %%
sql = loh.main(sub_date_end='2020-06-01', sample=1)

# %%
# df_ = bq.bq_query(sql)
# df = proc_df(df_)

# %%
df.iloc[:, :4]

# %%
dest = bq.BqLocation('wbeard_fission_test_dirp', dataset='analysis',
    project_id='moz-fx-data-shared-prod')
bq.bq_upload(df.iloc[:, :4], dest.no_tick)

# %%
f"""select * from {bq.bq_locs['input_data'].sql}"""

# %%
dfh = bq.bq_query(f"""select * from {bq.bq_locs['input_data'].sql}""")

# %%
dfh2 = bq.bq_query(loh.dl_agg_query())


# %% [markdown]
# # Raw histograms

# %%
# sql = mu.read("../fis/data/hist_data_proto.sql")
# df_ = bq_read(sql)

# %%
def hist2null(s):
    return s.map(len) == 0


# all_keys = get_all_keys(df.cycle_collector_max_pause)
# k2ix = {k: i for i, k in enumerate(all_keys)}
# ix2k = {v: k for k, v in k2ix.items()}
# # k2ix

# %%
d = {2: 1, 6: 1, 7: 1}

k2ix = typed_dict(k2ix)
d = typed_dict(d)

arr = np.zeros(48, np.int64)
dict2array(nd, k2ix, arr)

# %%
import attr
from fis.models import hist

# s = df.pipe(lambda df: df[df.cycle_collector.map(len) > 0]).cycle_collector
# h = hist.Hist(s)

_df = df.pipe(lambda df: df[df.cycle_collector.map(len) > 0])
d1 = _df.query("br == 'enabled'")
d2 = _df.query("br == 'disabled'")

s1 = d1.cycle_collector
s2 = d2.cycle_collector

h1 = hist.Hist(s1)
h2 = hist.Hist(s2)

d1 = _df.query("date == '2020-06-25'")
hb1 = d1.query("br == 'enabled'").pipe(lambda x: hist.Hist(x.cycle_collector))
hb2 = d1.query("br == 'disabled'").pipe(lambda x: hist.Hist(x.cycle_collector))

# %%
_h = hist.Hist(s1[:4])

# %%
_h.p_arr

# %% [markdown]
# ### Histogram playground

# %%
s1 = d1.cycle_collector
h1 = hist.Hist(s1)


# %%
hb2.dir_alpha_all_sqrt
# .sum(axis=1)

# %% [markdown]
# ### Square root?

# %%
nr_ = 1; nc = 3
all_splts, spi = mu.mk_sublots(nrows=nr_, ncols=nc, figsize=(nc * 8, nr_ * 5), sharex=False)

a = hb1.p_arr.sum(axis=1)
spi.n
plt.hist(a, bins=100, density=1, alpha=.5);
spi.n
plt.hist(a ** .5, bins=100, density=1, alpha=.5);
spi.n
plt.hist(np.log(a), bins=100, density=1, alpha=.5);

# %%
dt.datetime.now() - pd.Timedelta(hours=18)


# %%
def plot_pareto(a):
    n = len(a)
    a = np.sort(a)[::-1]
    apct = (a / a.sum()).cumsum()
    plt.plot(np.linspace(0, 100, n), apct)


# %%
plot_pareto(a)
plot_pareto(np.sqrt(a))

# %%
# sampled_means = est_statistic(h1)
sampled_means = hist.est_statistic(h1.to_dict())
sampled_means

# %%
# sampled_means = est_statistic(h1)
sampled_means = hist.est_statistic(dict(zip(h1.ks, h1.dir_alpha_all_sqrt)))
sampled_means

# %%
/len h1.p_arr

# %%
_paggd = aggd.reset_index(drop=0)
_paggd[:3]

# %%
gmean = lambda x: sts.gmean(x + 1e-6)
hist.est_statistic(h1.to_dict(), stat_fn=gmean)


# %%
def agg_hist(s, stat_fn=gmean, fn=z.identity):
    h = hist.Hist(s)
    d = h.to_dict(fn=fn)
    return hist.est_statistic(
        d, n_hists=10_000, client_draws=10, stat_fn=stat_fn, ret_quantiles=[.05, .5, .95]
    )

agg_gmean = partial(agg_hist, stat_fn=gmean)
agg_gmean2 = partial(agg_hist, stat_fn=gmean, fn=np.sqrt)
agg_gmean3 = partial(agg_hist, stat_fn=gmean, fn=None)
agg_mean = partial(agg_hist, stat_fn=np.mean)
agg_med = partial(agg_hist, stat_fn=np.median)

# %%
# _agg_geo = _df.groupby(["date", "br"])[c.cycle_collector].apply(agg_gmean).unstack().fillna(0)
# _agg_geo2 = _df.groupby(["date", "br"])[c.cycle_collector].apply(agg_gmean2).unstack().fillna(0)
_agg_geo3 = _df.groupby(["date", "br"])[c.cycle_collector].apply(agg_gmean3).unstack().fillna(0)
# _agg_mean = _df.groupby(["date", "br"])[c.cycle_collector].apply(agg_mean).unstack().fillna(0)
# _agg_med = _df.groupby(["date", "br"])[c.cycle_collector].apply(agg_med).unstack().fillna(0)

# %%
r1 = vz.plot_errb(_agg_geo.reset_index(drop=0)) | vz.plot_errb(
    _agg_mean.reset_index(drop=0)
)
r2 = (
    vz.plot_errb(_agg_geo3.reset_index(drop=0)).properties(title="Unnormalized--")
    | vz.plot_errb(_agg_geo.reset_index(drop=0)).properties(
        title="Normalize-all clients same weight"
    )
    | vz.plot_errb(_agg_geo2.reset_index(drop=0)).properties(title="Normalize-Sqrt")
)
r2.resolve_scale(y="shared")


# %%
class Branches:
    def __init__(self, aggd):
        self.en = aggd.xs('enabled', level=-1).reset_index(drop=0)
        self.dis = aggd.xs('disabled', level=-1).reset_index(drop=0)
    
    def pat(self):
        p1 = self.en.pat(x='date', y='p50', e1='p05', e2='p95', st='-', nz=0)
        p2 = self.dis.pat(x='date', y='p50', e1='p05', e2='p95', st='-', nz=0)
        return p1 + p2


# %%
vz.plot_errb(_agg_geo.reset_index(drop=0)) | vz.plot_errb(_agg_med.reset_index(drop=0))

# %%

# %%
vz.plot_errb(_paggd)

# %%
est_statistic(h2, stat_fn=gmean)

# %%
est_statistic(h2)

# %% jupyter={"outputs_hidden": true}
n_hists = 10
client_draws = 100
hists3 = nr.dirichlet(hb2.dir_alpha_all, r)
for hist in hists3:
    samps = nr.choice(k, p=hist, n=client_draws)


# %%
plt.plot(hb1.dir_alpha_all)
# plt.plot(hb2.dir_alpha_all)

# %%
plt.plot(hb1.dir_alpha_all_sqrt)

# %%
plt.plot(h1.dir_alpha_all)
plt.plot(h2.dir_alpha_all)

# %%
print(h1.dir_alpha_all)
print(h2.dir_alpha_all)

# %%
h1.dir_alpha_all.sum()

# %%
samps1 = h1.sample_geo_means(10, 1_000)
samps2 = h2.sample_geo_means(10, 1_000)


# %%
def agg_hists(hs, qs=[.1, .5, .9]):
    hs = hs[hs.map(len) > 0]
    h1 = hist.Hist(hs)
    r = h1.sample_geo_means(10, 1_000)
    return dict(zip('p10 p50 p90'.split(), np.quantile(r, qs)))

agg_hists(s1, qs=[.05, .5, .95])

# %%
samp_df_dir = (
    df.groupby([c.br, c.date])
    .cycle_collector.apply(z.comp(agg_hists))
    .unstack()
    .fillna(0)
    .reset_index(drop=0)
)

# %% jupyter={"outputs_hidden": true}
samp_df_dir


# %%

# %%
def _pl(samp_df_dir):
    color = 'br'
    x = "date"
    y = 'p50'

    h = Chart(samp_df_dir).mark_line().encode(
        x=A.X(x, title=x),
        y=A.Y(y, title=y, scale=A.Scale(zero=False)),
        color=color,
        tooltip=[color, x, y]
    )
    err = h.mark_errorband().encode(y='p10', y2='p90')
    return h + err

    return (h + h.mark_point()).interactive()

_pl(samp_df_dir)

# %% jupyter={"outputs_hidden": true}
# gb.apply?

# %%

# %%
plt.hist(samps1, bins=10, density=1, alpha=1)
plt.hist(samps2, bins=10, density=1, alpha=.1)
None

# %%
df.br.unique()

# %%
hi = hist.sample_arr(100, 30, h.p_norm, h.ks)
hi

# %%
hi

# %%
# dir_alpha = prob2dir_alpha(h.p_arr)
# dir_alpha_all = h.p_arr.sum(axis=0)

p = h.p_arr
p


# %%

# %%
# @njit
def samp_dir(dir_alpha, n_dists):
    m = nr.dirichlet(dir_alpha, size=n_dists)
    return gmean(m)


# %%
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


def enumerate_hists(counts=h.p_arr, ks=h.ks, eps=1e-6):
    enumerated_vals = [
        e
        for hist in counts
        for val, ct in zip(ks, hist)
        for e in [val] * ct
    ]

    enumerated_vals = np.array(enumerated_vals)
    return sts.gmean(enumerated_vals + eps)

def test_gmean_hist2d(counts=h.p_arr, ks=h.ks, eps=1e-6):
    est1 = enumerate_hists(counts, ks, eps)
    est2 = gmean_hist2d(h.p_arr, np.array(h.ks), eps=eps)
    assert est1 == est1


# %%
def sample_geo_means(dir_params, ks, n_hists):
    ks = np.asarray(ks)
    hists = nr.dirichlet(dir_params, size=n_hists)    
    return gmean_hist2d(hists, ks, eps=1e-6)

def sample_geo_means_rep(dir_params, ks, n_hists, n_reps):
    return np.array([sample_geo_means(dir_params, ks, n_hists) for _ in range(n_reps)])

# sample_geo_means(dir_alpha_all, h.ks, 100)

samp_geo_means = sample_geo_means_rep(h.dir_alpha_all, h.ks, 100, 1_000)
np.percentile(samp_geo_means, [5, 50, 95])

# %%
samp_geo_means = sample_geo_means_rep(h1.dir_alpha_all, h.ks, 100, 1_000)
np.percentile(samp_geo_means, [5, 50, 95])

# %%
samp_geo_means = sample_geo_means_rep(h2.dir_alpha_all, h.ks, 100, 1_000)
np.percentile(samp_geo_means, [5, 50, 95])

# %%
plt.hist(samp_geo_means, bins=100, density=1, alpha=.5)
None

# %%
# enumerate_hists(counts=h.p_arr, ks=h.ks, eps=1e-6)
# test_gmean_hist2d(counts=h.p_arr, ks=h.ks, eps=1e-6)

gmean_hist1d(h.p_arr[0], np.log(np.array(h.ks) + 1e-6))

# %%
gmean_hist2d(h.p_arr, np.array(h.ks))

# %%
# %timeit enumerate_hists(counts=h.p_arr, ks=h.ks, eps=1e-6)
# %timeit gmean_hist2d(h.p_arr, np.array(h.ks))

# %%
d = (
    DataFrame(
        dict(
            c=h.p_arr[0],
            k=h.ks
            #                    np.log(np.array(h.ks) + 1e6),
        )
    )
    .query("c > 0")
    .assign(lk=lambda df: np.log(df.k + 1e-6))
)
mu = d.lk.mean()
print(mu)
print(np.exp(mu))
d


# %%
def gmean_s(s):
    s = s.values
    return sts.gmean(s + 1) - 1

@njit
def gmean(a):
    return sts.gmean(a + 1) - 1

# sts.gmean()

# samp_dir(dir_alpha_all, 5, )


# %%
nr.dirichlet(dir_alpha_all, size=1)

# %%
dir_alpha[0]

# %%
dir_alpha


# %%
def build_dist_arr(s):
    all_keys = get_all_keys(s)


# all_keys

# %%
df.cycle_collector[0]

# %%
df.unq_tabs_mean[0]


# %%
df

# %% [markdown]
# # Just the means

# %%
sql = mu.read("../fis/data/hist_data_means_proto.sql")
dfm_ = bq_read(sql)
