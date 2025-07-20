import matplotlib.pyplot as plt
import io, base64

def make_scatterplot(df, x, y, slope, intercept, max_kb=100):
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y])
    xs = df[x].astype(float)
    ax.plot(xs, slope*xs + intercept, linestyle="--", color="red")
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    data = buf.getvalue()
    if len(data) > max_kb*1024:
        raise ValueError("Plot exceeds size limit")
    return "data:image/png;base64," + base64.b64encode(data).decode()
