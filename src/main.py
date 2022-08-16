import util
import vis
import sys
import random
import numpy as np
from agents.basic import Random, Top, Bottom
from game.boolean.functional import (
    binary_data,
    n_ary_data,
    get_ssr_data,
    AND,
    OR,
    XOR,
    NAND,
    IMPLIES,
    IFF,
)
from game.boolean.signaltree import SignalTree, get_optimal_ssr
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

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/main.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    # load game settings
    config_fn = sys.argv[1]
    configs = util.load_configs(config_fn)

    num_rounds = configs["num_rounds"]
    input_size = configs["input_size"]
    learning_rate = configs["learning_rate"]
    random_seed = configs["random_seed"]
    save_accuracy_plot = configs["file_paths"]["save_accuracy_plot"]
    save_languages_fn = configs["file_paths"]["save_languages"]

    util.set_seed(random_seed)

    # define learning problem
    # dataset = binary_data()
    # dataset = n_ary_data(n=input_size, connective=AND)
    dataset = get_ssr_data(f=XOR)
    for example in dataset:
        for k, v in example.items():
            print(k, v)

    # initialize network and parameters
    # net = SignalTree(input_size=input_size, learning_rate=learning_rate)
    net = get_optimal_ssr()
    # net = Random()
    # net = Bottom()

    accuracy = []
    # main training loop
    for r in tqdm(range(num_rounds)):
        example = np.random.choice(dataset)

        x = example["input"]

        print("VALUE of x: ", x)
        # x = list(example["input"])  # copy and shuffle order
        # random.shuffle(x)

        y = example["label"]
        y_hat = net(x)

        # print(f"train mode set to {net.train_mode}")
        net.update(reward_amount=int(y == y_hat))

        acc = empirical_accuracy(net, dataset, num_rounds=100)
        # record accuracy
        if r % 10 == 0:
            print(f"Accuracy on round {r}: {round(acc, 2)}")

        accuracy.append(acc)

    # analysis
    vis.plot_accuracy(save_accuracy_plot, accuracy)

    # Inspect languages for the optimal ssr net
    sender_layer, receiver = net.layers
    sender_a, sender_b = sender_layer.agents
    # sender_a = sender_a.signaler
    # sender_b = sender_b.signaler
    # receiver = receiver.receiver
    languages = [
        sender_a.to_language(data={"name": "Sender A"}, threshold=0.5),
        sender_b.to_language(data={"name": "Sender B"}, threshold=0.5),
        receiver.to_language(data={"name": "Receiver"}, threshold=0.5),
    ]

    util.save_languages(fn=save_languages_fn, languages=languages)


if __name__ == "__main__":
    main()
