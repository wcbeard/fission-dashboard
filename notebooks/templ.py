# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook ("") tries to download all recent experiments via dbx

# %%
from boot_utes import add_path, path, reload, run_magics
from matplotlib import MatplotlibDeprecationWarning

add_path('..', '../fis/', '~/repos/myutils/', )
add_path("/Users/wbeard/repos/dscontrib-moz/src/")
import dscontrib.wbeard as dwb

from utils.fis_imps import *; exec(pu.DFCols_str); exec(pu.qexpr_str); run_magics()
# import utils.en_utils as eu; import data.load_data as ld; exec(eu.sort_dfs_str)

sns.set_style("whitegrid")
A.data_transformers.enable("json", prefix="data/altair-data")
S = Series
D = DataFrame
# %%

def pipe(h, f):
    return f(h)

A.Chart.pipe = pipe

def add_point(h):
    return h + h.mark_point()

def add_line(h):
    return h + h.mark_line()

nz = A.Scale(zero=False)
lgs = A.Scale(type="log", zero=False)

# %% [markdown]
# # Load
