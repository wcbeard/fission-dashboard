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

sns.set_style("whitegrid")
S = Series
D = DataFrame
# %%
sql = mu.read("../fis/data/hist_data_means_proto.sql")
dfm_ = bq_read(sql)

# %%
from numba import typed


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


def arr_of_str2dict_(arr):
    """
    [{'key': 0, 'value': 0}, {'key': 1, 'value': 1}] -> {1: 1}
    """
    return {d["key"]: d["value"] for d in arr if d["value"]}


arr_of_str2dict = z.compose(typed_dict, arr_of_str2dict_)
hist_cols = [
    "unq_tabs",
    "unq_sites_per_doc",
    "cycle_collector_slice_during_idle",
    "gc_slice_during_idle",
    "cycle_collector",
    "cycle_collector_max_pause",
    "gc_max_pause_ms_2",
    "gc_ms",
]

# %%
dfm = dfm_.copy()

# %%
# plt.figure(figsize=(16, 6)); ax = plt.gca()

dfm[hist_cols].hist(bins=100, density=1, alpha=.5, figsize=(16, 6))

# %%
dfm[hist_cols].add(1).pipe(np.log10).hist(bins=100, density=1, alpha=.5, figsize=(16, 6))
None


# %%
def gmean(s):
    s = s.values
    return sts.gmean(s + 1) - 1

sts.gmean()

# %%
pd.__version__


# %%
def gmean(s):
    s = s.values
    return sts.gmean(s + 1) - 1
dfm['cycle_collector'].agg([gmean, 'mean', 'median'])

# %%
dfm.cycle_collector.sample(1_000, replace=True)

# %%
dfm.cycle_collector.add(1).pipe(np.log10).hist(bins=100, density=1, alpha=.5)
