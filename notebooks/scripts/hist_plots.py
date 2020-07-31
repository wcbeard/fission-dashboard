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
    "..", "/Users/wbeard/repos/fis/fis/", "/Users/wbeard/repos/fis/", "~/repos/myutils/",
)
add_path("/Users/wbeard/repos/dscontrib-moz/src/")

# %%
from matplotlib import MatplotlibDeprecationWarning

import sys
print(sys.path)

# import ipdb; ipdb.set_trace();
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

import data.load_crap as lc
import dscontrib.wbeard.altair_utils as aau

aau.set_ds(A)

DataFrame.pat = aau.pat

# %%
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

# %% [markdown]
# # ETL

# %%
from models import hist
import data.load_agg_hists as loh
from utils import bq

# %%
list_of_docs_to_dict = lambda l: {d["key"]: d["value"] for d in l}

# %%
dfh2_ = bq.bq_query(loh.dl_agg_query())

# %%
txf_fns = {
    h: lambda df: df[h].map(list_of_docs_to_dict)
    for h in loh.hist_cols
}
dfh2 = dfh2_.assign(**txf_fns).assign(date=lambda df: pd.to_datetime(df.date))

# %% [markdown]
# # Plot Hists

# %%
s = dfh2.unq_sites_per_doc
h = s[0]

# %%

# hist.est_statistic(h1.to_dict(), stat_fn=gmean)

# %%
gmean = lambda x: sts.gmean(x + 1e-6)

agg_gmean = partial(hist.est_statistic, n_hists=10_000, client_draws=10, stat_fn=gmean, ret_quantiles=[0.05, 0.5, 0.95])
# agg_gmean = lru_cache()(agg_gmean)
# hist.est_statistic(
#     h, n_hists=10_000, client_draws=10, stat_fn=gmean, ret_quantiles=[0.05, 0.5, 0.95]
# )



# %%
df = DataFrame({'x': [1, 2, 3]})

def _pl(pdf):
    x = "x"
    y = 'x'

    h = (
        A.Chart(pdf)
        .mark_point()
        .encode(
            x=A.X(x, title=x),
            y=A.Y(y, title=y, scale=A.Scale(zero=False)),
            # color=color,
            tooltip=[
                # color,
                x,
                y,
            ],
        )
    )

    return h.interactive()

A.data_transformers.enable('default')

_pl(df)

# %% [markdown]
# def cache_dict(f):
#     @lru_cache()
#     def tup_f(tup):
#         d = dict(tup)
#         return f(d)
#         
#     @wraps(f)
#     def dict_f(dict_arg):
#         tuple_arg = tuple(sorted(dict_arg.items()))
#         return tup_f(tuple_arg)
#         
#     return dict_f
#
# agg_gmean_cache = cache_dict(agg_gmean)

# %% [markdown]
# agg_gmean_cache(h)

# %% [markdown]
# def summarize_hist_df(df, hist_srs):
#     ps = [agg_gmean_cache(h) for h in hist_srs]
#     df = pd.concat([df[["date", "br", "n_cid"]], DataFrame(ps)], axis=1)
#     return df
#
#
# hdf = summarize_hist_df(dfh2, s)

# %% [markdown]
# # with A.data_transformers.enable('json'):
# #     A.Chart.save(hh, '../reports/figures/.png')
#
# # aau.set_json(A=A)
#
# # A.data_transformers.enable('json')
# A.data_transformers.enable('default')
# vz.plot_errb(hdf).properties(height=200, width=1400, title=s.name)
