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
# %% [markdown]
# # Load

# %%
s = """
() AS ("{\"buck
"""
# .replace('\"', 'q')
print(s)
print(s.replace('\"', 'q'))


# %%
q = '''
-- `mozfun.hist.merge`(ARRAY_AGG(`mozfun.hist.extract`(h)))
-- hmerge(ARRAY_AGG(hist(h)))
CREATE TEMP FUNCTION hist_(h ANY TYPE) AS (`mozfun.hist.extract`(h));
CREATE TEMP FUNCTION hist100(h ANY TYPE) AS (`mozfun.hist.extract`(coalesce(h, null_hist_str100())));
CREATE TEMP FUNCTION hist1e4(h ANY TYPE) AS (`mozfun.hist.extract`(coalesce(h, null_hist_str10000())));
CREATE TEMP FUNCTION hmerge(h ANY TYPE) AS (`mozfun.hist.merge`(h));
CREATE TEMP FUNCTION get_key(hist ANY TYPE, k string) AS (`mozfun.map.get_key`(hist, k));

-- CREATE TEMP FUNCTION null_hist_str() AS ("{\\"bucket_count\\":50,\\"histogram_type\\":0,\\"sum\\":0}");
CREATE TEMP FUNCTION null_hist_str100() AS ("{\\"bucket_count\\":50,\\"histogram_type\\":0,\\"sum\\":0,\\"range\\":[1,100]}");
CREATE TEMP FUNCTION null_hist_str10000() AS ("{\\"bucket_count\\":50,\\"histogram_type\\":0,\\"sum\\":0,\\"range\\":[1,10000]}");

CREATE TEMP FUNCTION hist_str_to_mean(hist ANY TYPE) AS (
  `moz-fx-data-shared-prod.udf.histogram_to_mean`(`moz-fx-data-shared-prod.udf.json_extract_histogram`(hist))
);


/*
unq_tabs: {"bucket_count":50,"histogram_type":0,"sum":2,"range":[1,100],"values":{"0":0,"1":2,"2":0}}
*/
CREATE TEMP FUNCTION major_vers(st string) AS (
  -- '10.0' => 10
  cast(regexp_extract(st, '(\\\\d+)\\\\.?') as int64)
);


with base as (
select
  m.client_id,
  submission_timestamp as ts,
  
  m.payload.histograms.fx_number_of_unique_site_origins_all_tabs as unq_tabs,
  m.payload.histograms.fx_number_of_unique_site_origins_per_document as unq_sites_per_doc,
  m.payload.histograms.cycle_collector as cycle_collector,
  m.payload.histograms.cycle_collector_max_pause as cycle_collector_max_pause,
  m.payload.histograms.cycle_collector_slice_during_idle as cycle_collector_slice_during_idle,
  -- cycle_collector_slice_during_idle
  m.payload.histograms.gc_max_pause_ms_2 as gc_max_pause_ms_2,
  m.payload.histograms.gc_ms as gc_ms,
  m.payload.histograms.gc_slice_during_idle as gc_slice_during_idle,
  
  coalesce(m.environment.system.gfx.features.wr_qualified.status, '') = 'available' as wr_av,
  get_key(m.environment.experiments, 'bug-1622934-pref-webrender-continued-v2-nightly-only-nightly-76-80').branch is null as no_wr_exp,
  sample_id,
from `moz-fx-data-shared-prod.telemetry.main` m
where
  date(m.submission_timestamp) between '2020-06-21' and '2020-06-21'
  and m.normalized_channel = 'nightly'
  and m.normalized_app_name = 'Firefox'
  and sample_id between 1 and 1
--   and major_vers(m.normalized_os_version) = 10
)

, b1 as (
select
  client_id,
  unq_tabs as hu,
  cycle_collector as cc,
--   hist(null_hist_str100()) as nh,
--   cycle_collector,
--   null_hist_str100() as nhs,

--   count(*) over (partition by client_id) as cid_n,
--   dense_rank() over (order by client_id) as cid,

--   coalesce(unq_tabs, ""),
--   hist(coalesce(unq_tabs_mean, "")),
--   hmerge(ARRAY_AGG(hist(cycle_collector))) as cycle_collector,
--   hist(unq_tabs),
--   hist(cycle_collector) as cc,
--   hmerge(ARRAY_AGG(hist(unq_tabs))) as unq_tabs_mean
from base
where
  client_id = '01308377-38f1-46d7-b008-ee0f2fc90164'
)


, b2 as (
select
  client_id,
  hist100(null_hist_str100()) as nh,

  count(*) over (partition by client_id) as cid_n,
  dense_rank() over (order by client_id) as cid,
--   coalesce(unq_tabs, ""),
--   hist(coalesce(unq_tabs_mean, "")),
  hmerge(ARRAY_AGG(hist100(cycle_collector))).values as cycle_collector,
  hmerge(ARRAY_AGG(hist100(unq_tabs))).values as unq_tabs,
from base
where
  client_id = '01308377-38f1-46d7-b008-ee0f2fc90164'
group by 1
)



select
  *,
from b2
-- where
--   cid_n > 1
'''

# %%

# %%
sql = mu.read('../fis/data/hist_data_proto.sql')
df_ = bq_read(sql)

# %%

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
h_kw = {h: lambda df, h=h: df[h].map(arr_of_str2dict) for h in hist_cols}
df = df_.assign(**h_kw)

# %% [markdown]
# # Raw histograms

# %%
len(df)

# %%
s = df_.cycle_collector[0]
s

# %%
arr_of_str2dict(s)

# %%
df[:3]

# %%
/list df


# %%
def hist2null(s):
    return s.map(len) == 0


get_all_keys = lambda srs: sorted({k for doc in srs for k in doc})
all_keys = get_all_keys(df.cycle_collector_max_pause)
k2ix = {k: i for i, k in enumerate(all_keys)}
ix2k = {v: k for k, v in k2ix.items()}
# k2ix

# %%
len(all_keys)

# %%
d = {2: 1, 6: 1, 7: 1}

k2ix = typed_dict(k2ix)
d = typed_dict(d)

arr = np.zeros(48, np.int64)
dict2array(nd, k2ix, arr)

# %%
del k2ix, all_keys, ix2k

# %%
from fis.models import hist
import attr

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
    

class Hist:
    def __init__(self, s):
        self.name = s.name
        
        self.all_keys = get_all_keys(s)
        self.k2ix = k2ix= typed_dict({k: i for i, k in enumerate(self.all_keys)})
        self.ix2k = ix2k = typed_dict({v: k for k, v in self.k2ix.items()})
        self.ks = sorted(k2ix)
        
        self.dct_lst = ds = typed_list(s)
        self.p_arr = p = dicts2arrays(ds, k2ix)
        self.p_norm = p / p.sum(axis=1)[:, None]

    
# ds = typed_list([d, d])
# dicts2arrays(ds, k2ix)

s = df.pipe(lambda df: df[df.cycle_collector.map(len) > 0]).cycle_collector
h = Hist(s)

# %%

    
hi = hist.sample_arr(100, 30, h.p_norm, h.ks)
hi

# %%
df.cycle_collector

# %%
len(df)

# %%
len(h.p_norm)

# %%
h.ix2k[4]

# %%
nr.choice(h.ks, size=40, p=hi)

# %%
h.ks

# %%
p = h.p_arr
p


# %%
def prob2dir_alpha(p):
    pp1 = p + 1
    sa = pp1.sum(axis=1)[:, None]

    return pp1 / sa

dir_alpha = prob2dir_alpha(h.p_arr)

# %%
dir_alpha[0]

# %%
_dir_draw = nr.dirichlet(dir_alpha.T, 2)
np.round(_dir_draw, 2)

# %%
p / p.sum(axis=1)

# %%
nr.dirichlet([
    [  0,   0,   1,   0,   0,   0,   1,   1,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   1,   0,   0,   0,   1,   1,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
])

# %%
nr.choice(h.ks, p=h.p_arr)

# %%
h.p_arr

# %%
# ix2k

# %%
s = df.cycle_collector[:5].pipe(typed_list)
s

# %%
dicts2arrays(s, k2ix)


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
sql = mu.read('../fis/data/hist_data_means_proto.sql')
dfm_ = bq_read(sql)
