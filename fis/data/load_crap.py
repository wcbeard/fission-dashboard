from numba import typed  # type: ignore
import numpy as np  # type: ignore
import toolz.curried as z  # type: ignore
import pandas as pd  # type: ignore


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


def get_hist_cols_raw_dl(df):
    return (
        df.loc[0]
        .map(lambda x: type(x) == np.ndarray)
        .pipe(lambda x: x[x])
        .index.tolist()
    )


def get_hist_cols(df):
    return (
        df.loc[0].map(type).pipe(lambda x: x[x == typed.Dict]).index.tolist()
    )


def get_dict_hist_cols(df):
    return (
        df.loc[0]
        .map(lambda x: type(x) == dict)
        .pipe(lambda x: x[x])
        .index.tolist()
    )


def pr_name(s):
    print(s.name)
    return s


def read(fname):
    with open(fname, "r") as f:
        return f.read()


def load(
    refresh,
    bq_read,
    sqlfn="../fis/data/hist_data_proto.sql",
    dest="/tmp/hists.pq",
):
    if refresh:
        sql = read(sqlfn)
        df_ = bq_read(sql)
        hist_cols = get_hist_cols_raw_dl(df_)
        h_kw = {
            h: lambda df, h=h: df[h].map(arr_of_str2dict) for h in hist_cols
        }
        df_ = df_.assign(**h_kw)
        # hist_cols = get_hist_cols(df)
        hist_cols_asn = {
            c: lambda x, c=c: x[c].map(z.keymap(str)) for c in hist_cols
        }
        ds = df_.assign(**hist_cols_asn)
        ds.to_parquet(dest)

    ds = pd.read_parquet(dest)
    # print(ds.cycle_collector)
    # return ds

    fn = z.compose(
        typed_dict, z.keymap(int), z.valfilter(lambda x: x is not None)
    )
    hist_cols = get_dict_hist_cols(ds)
    hist_cols_asn = {
        c: lambda df, c=c: df[c].map(fn) for c in hist_cols
    }
    #     return ds
    df = ds.assign(**hist_cols_asn)
    return df
