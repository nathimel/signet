from game import binary_data, empirical_accuracy, n_ary_data
import util
import vis
import sys
import numpy as np
from network import SignalTree


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

    util.set_seed(random_seed)

    # define learning problem
    # dataset = binary_data()
    dataset = n_ary_data(n=input_size)

    # initialize network and parameters
    net = SignalTree(input_size=input_size)
    amount = learning_rate * 1.0

    accuracy = []
    # main training loop
    for r in range(num_rounds):
        example = np.random.choice(dataset)
        x = example["input"]
        y = example["label"]
        y_hat = net(x)

        if y == y_hat:
            net.reward(amount)
        else:
            net.punish(amount)

        acc = empirical_accuracy(net, dataset)
        # record accuracy
        if r % 10 == 0:
            print(f"Accuracy on round {r}: {round(acc, 2)}")

        accuracy.append(acc)

    # analysis
    vis.plot_accuracy(save_accuracy_plot, accuracy)


if __name__ == "__main__":
    main()
