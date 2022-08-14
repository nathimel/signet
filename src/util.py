import argparse
import yaml
import numpy as np
import random
from plotnine import ggplot
from languages import SignalingLanguage


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


def save_languages(fn: str, languages: list[SignalingLanguage]) -> None:
    """Save a list of languages to a YAML file."""
    data = {"languages": list(lang.yaml_rep() for lang in languages)}
    with open(fn, "w") as outfile:
        yaml.safe_dump(data, outfile)
