import plotnine as pn
import pandas as pd
import numpy as np
from util import save_plot


def plot_accuracy(fn: str, accuracies: list[float]) -> None:
    """Construct and save a basic plotnine line plot of accuracy vs rounds."""
    data = pd.DataFrame(
        data={
            "accuracy": accuracies,
            "round": [np.log10(x) for x in range(len(accuracies))],
        }
    )
    # Set data and the axes
    plot = (
        pn.ggplot(data=data, mapping=pn.aes(x="round", y="accuracy"))
        + pn.geom_line(size=1, data=data)
        + pn.ylim(0, 1)
        + pn.xlab("Rounds (log)")
        + pn.ylab("Accuracy")
        + pn.scale_color_cmap("cividis")
    )
    save_plot(fn, plot)
