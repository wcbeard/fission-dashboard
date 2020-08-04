import altair as A
import matplotlib.pyplot as plt
import seaborn as sns


def plot_errb(pdf, ytitle="geomean", zero=True):
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
            y=A.Y(y, title=ytitle, scale=A.Scale(zero=zero)),
            color=color,
            tooltip=[color, x, y],
        )
    )
    herr = h.mark_errorband().encode(
        y=A.Y(lb, title=ytitle), y2=A.Y2(ub, title=ytitle)
    )

    return (h + herr).interactive()


def plot_multimodal(df):
    plt.figure(figsize=(16, 6))
    # ax = plt.gca()
    sns.violinplot(
        x="date",
        y="samps",
        data=df,
        hue="br",
        split=True,
        order=sorted(df.date.unique()),
        inner="quart",
    )
    sns.despine(left=True)
    plt.xticks(rotation=75)
    plt.legend(loc="lower left")
