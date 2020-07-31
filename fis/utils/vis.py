import altair as A


def plot_errb(pdf, ytitle="geomean"):
    color = "br"
    x = "date"
    y = "p50"

    lb = "p05"
    ub = "p95"

    h = (
        A.Chart(pdf)
        .mark_line()
        .encode(
            x=A.X(x, title=x),
            y=A.Y(y, title=ytitle, scale=A.Scale(zero=True)),
            color=color,
            tooltip=[color, x, y],
        )
    )
    herr = h.mark_errorband().encode(
        y=A.Y(lb, title=ytitle), y2=A.Y2(ub, title=ytitle)
    )

    return (h + herr).interactive()
