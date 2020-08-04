import pandas as pd


def to_date_col(colname):
    """
    import datetime as dt
    df.assign(d=to_date_col('d'))
    type(df.d.iloc[0]) == dt.date
    """

    def fn(df):
        return pd.to_datetime(df[colname]).dt.date

    return fn


def rn_prob_col(x):
    ".05 -> p05"
    if isinstance(x, int) and (x not in (0, 1)):
        return x
    if not isinstance(x, float):
        return x
    return f"p{int(x * 100):02}"


class AttrDict(dict):
    """
    Dict with attribute access of keys
    http://stackoverflow.com/a/14620633/386279
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        d = super().copy()
        return AttrDict(d)


def s(x, thresh=.08):
    ss = pd.Series(x).sort_index()
    p = ss / ss.sum()
    return ss[p > thresh]
