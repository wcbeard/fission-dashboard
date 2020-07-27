def rn_prob_col(x):
    ".05 -> p05"
    if isinstance(x, int) and (x not in (0, 1)):
        return x
    if not isinstance(x, float):
        return x
    return f"p{int(x * 100):02}"
