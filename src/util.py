import argparse
import yaml
import numpy as np
import random
from plotnine import ggplot


def set_seed(seed: int) -> None:
    """Sets various random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def save_plot(fn: str, plot: ggplot, width=10, height=10, dpi=300) -> None:
    """Save a plot with some default settings."""
    plot.save(fn, width=10, height=10, dpi=300)


def load_configs(fn: str) -> dict:
    """Load the configs .yml file as a dict."""
    with open(fn, "r") as stream:
        configs = yaml.safe_load(stream)
    return configs
