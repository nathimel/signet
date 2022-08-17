from game.glass.glasses import GlassTree
import util
import vis
import sys
import numpy as np
from agents.basic import Random, Top, Bottom
from game.boolean.functional import (
    n_ary_data,
    AND,
    OR,
    XOR,
    NAND,
    IMPLIES,
    IFF,
)
from game.boolean.signaltree import SignalTree
from languages import Signal, State
from tqdm import tqdm
from typing import Any


def empirical_accuracy(
    net: SignalTree,
    dataset: list[dict[str, Any]],
    num_rounds: int = None,
) -> float:
    """
    Evaluate the accuracy of a signaling networks by computing the average accuracy on the dataset.

    Args:
        num_rounds: an int representing how many interactions to record.
    """
    net.test()

    if num_rounds is None:
        num_rounds = len(dataset)

    num_correct = 0
    for _ in range(num_rounds):
        example = np.random.choice(dataset)
        x = example["input"]
        y = example["label"]
        y_hat = net(x)
        num_correct += 1 if y_hat == y else 0

    net.train()
    return num_correct / num_rounds


criterion = lambda prediction, label: int(prediction == label)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/main.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    # load game settings
    config_fn = sys.argv[1]
    configs = util.load_configs(config_fn)

    num_rounds = int(configs["num_rounds"])
    input_size = configs["input_size"]
    learning_rate = configs["learning_rate"]
    random_seed = configs["random_seed"]
    save_accuracy_plot = configs["file_paths"]["save_accuracy_plot"]

    util.set_seed(random_seed)

    dataset = n_ary_data(
        n=input_size,
        connective=XOR,
        input_type=Signal,  # we assume inputs are already signals
        output_type=State,  # and that outputs are acts (states)
    )

    # initialize network and parameters
    net = SignalTree(
        input_size=input_size,
        learning_rate=learning_rate,
    )

    accuracy = []
    # main training loop
    for r in tqdm(range(num_rounds)):
        example = np.random.choice(dataset)

        x = example["input"]
        y = example["label"]
        y_hat = net(x)

        net.update(reward_amount=criterion(y_hat, y))

        acc = empirical_accuracy(net, dataset, num_rounds=100)
        # record accuracy
        if r % 100 == 0:
            print(f"Accuracy on round {r}: {round(acc, 2)}")

        accuracy.append(acc)

    # # analysis
    vis.plot_accuracy(save_accuracy_plot, accuracy)


if __name__ == "__main__":
    main()
