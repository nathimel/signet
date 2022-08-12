import plotnine as pn
import pandas as pd
from util import save_plot


def plot_accuracy(fn: str, accuracies: list[float]) -> None:
    """Construct and save a basic plotnine line plot of accuracy vs rounds."""
    data = pd.DataFrame(
        data={
            "accuracy": accuracies,
            "round": range(len(accuracies)),
        }
    )
    # Set data and the axes
    plot = (
        pn.ggplot(data=data, mapping=pn.aes(x="round", y="accuracy"))
        + pn.geom_line(size=1, data=data)
        + pn.ylim(0, 1)
        + pn.xlab("Rounds")
        + pn.ylab("Accuracy")
        + pn.scale_color_cmap("cividis")
    )
    save_plot(fn, plot)
