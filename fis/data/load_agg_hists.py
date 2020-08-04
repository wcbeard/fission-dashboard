import json

from fire import Fire
import pandas as pd

from fis.utils import bq
from fis import app_dir


def read_sql(fn="agg_hists.templ") -> str:
    with open(app_dir / "data" / fn, "r") as fp:
        templ = fp.read()
    return templ


unimodal_histograms = [
    "unq_tabs",
    "unq_sites_per_doc",
    "cycle_collector",
    "cycle_collector_max_pause",
    "gc_max_pause_ms_2",
    "gc_ms",
]

multimodal_histograms = [
    "cycle_collector_slice_during_idle",
    "gc_slice_during_idle",
]

hist_cols = unimodal_histograms + multimodal_histograms


def transform_kval_json_str(st):
    """
    BQ + python transforms hist `values` attribute from
    {"k": v} => [{'key': k, 'value': v}].
    This undoes this.
    """
    d = json.loads(st)
    values = d.pop("values")
    values_dct = {str(d["key"]): d["value"] for d in values}
    d["values"] = values_dct
    return json.dumps(d)


def proc_df(df):
    df = df.copy()

    for c in hist_cols:
        df[c] = df[c].map(transform_kval_json_str)
        # .astype(str)
    return df


def dl_agg_query():
    sql = read_sql("dl_agg_hists.templ")
    h_cols = ",\n  ".join(f"vals({h}) as {h}" for h in hist_cols)
    return sql.format(hist_cols=h_cols)


def proc_hist_dl(df):
    def list_of_docs_to_dict(l):
        return {d["key"]: d["value"] + 1 for d in l}

    txf_fns = {
        h: lambda df, h=h: df[h].map(list_of_docs_to_dict) for h in hist_cols
    }
    return df.assign(**txf_fns).assign(date=lambda df: pd.to_datetime(df.date))


def main(
    sub_date_end,
    fill_yesterday=False,
    n_days=1,
    sub_date_start=None,
    sample=100,
    ret_sql=True,
):
    if fill_yesterday:
        raise NotImplementedError
    if sub_date_start is None:
        n_days_ago = n_days - 1
        sub_date_start = bq.subdate_sub(sub_date_end, days=n_days_ago)
    sql = read_sql().format(
        sub_date_start=sub_date_start, sub_date_end=sub_date_end, sample=sample
    )
    if ret_sql:
        return sql

    dest = bq.bq_locs["input_data"]
    # import ipdb; ipdb.set_trace()
    df = bq.bq_query(sql)
    df = proc_df(df)
    bq.bq_upload(df, dest.no_tick)
    return


if __name__ == "__main__":
    Fire(main)
