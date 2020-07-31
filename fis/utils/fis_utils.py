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
