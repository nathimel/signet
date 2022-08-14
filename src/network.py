import random
import numpy as np
from agents import (
    SignalingModule,
    InputSender,
    HiddenSignaler,
    OutputReceiver,
    Layer,
    Sequential,
)
from languages import State
from typing import Any, Callable

from agents import get_quaternary_sender, get_ssr_receiver

class SignalTree(Sequential):
    """Signaling network for learning functions from lists of states (e.g., truth values of propositions) to states (e.g., the truth value of a complex proposition).

    A SignalTree is a signaling network with one agent as the final output classifier.

    For the task of predicting the truth values of complex propositional logic sentences from a list of their atoms' truth values, a natural network topology when agents are limited to binary input is the sentence's (binary) syntactic tree representation.
    """

    def __init__(
        self,
        input_size: int,
        **kwargs,
    ) -> None:
        """Construct a signaling network to have a triangle shape, so that each layer doubles in size moving backwards, starting from the single output agent.

        The idea is that the optimal network is a binary tree, corresponding to a the syntactic tree representation of the propositional sentence with the input layer of agents as leaves and the output agent as the root.

        This may create 'redundant' nodes in hidden layers that would not exist in the minimal syntactic tree representation of a sentence with `input_size` atoms, when `input_size` is an odd number.
        """
        if input_size in [0, 1]:
            raise ValueError(
                f"A signaling network must consist of 2 or more agents. Received input size {input_size}"
            )
        # TODO: give each agent constructor kwargs so i can pass in the above args

        # construct input and output layers
        input_layer = Layer([InputSender(input_size) for _ in range(input_size)])
        hidden_layers = []
        output_layer = OutputReceiver()

        # construct hidden layers
        layer_sizes = get_layer_sizes(input_size)
        # remove input and output layers, because we've already constructed them
        layer_sizes = layer_sizes[1:-1]

        if layer_sizes:
            # populate each hidden layer with agents
            hidden_layers = Sequential(
                [
                    Layer([HiddenSignaler(size) for _ in range(size)])
                    for size in layer_sizes
                ]
            )
        super().__init__(
            layers=[
                layer
                for layer in [input_layer, hidden_layers, output_layer]
                if layer  # possibly empty list of hidden layers
            ]
        )


# define baseline SSR net


def get_optimal_ssr() -> SignalTree:
    """Hard code parameters of a Sequential module identical to a size 2 SignalTree."""

    # # hard code the attention params of InputSenders
    attn_a = np.array([1.0, 0.0])  # always look at first state
    # sender_a.attention_layer.parameters = attn_a

    attn_b = np.array([0.0, 1.0])  # always look at second state
    # sender_b.attention_layer.parameters = attn_b

    # hard code the attention params of OutputReceiver
    attn_1 = attn_a  # first signal always from first sender
    attn_2 = attn_b  # second signal always from second sender

    # receiver.compressor.attention_1.parameters = attn_1
    # receiver.compressor.attention_2.parameters = attn_2

    # freeze these params only
    # sender_a.attention_layer.freeze()
    # sender_b.attention_layer.freeze()
    # receiver.compressor.freeze()


    sender_a = get_quaternary_sender()
    sender_b = get_quaternary_sender()
    sender_layer = Layer(agents=[sender_a, sender_b])
    receiver = get_ssr_receiver()
    net = Sequential(
        layers=[sender_layer, receiver]
    )

    return net


##############################################################################
# Utility functions
##############################################################################


def build_triangle_network(
    input_size: int = 2, **kwargs
) -> dict[str, list[SignalingModule]]:
    """Construct a signaling network to have a triangle shape, so that each layer doubles in size moving backwards, starting from the single output agent.

    The idea is that the optimal network is a binary tree, corresponding to a the syntactic tree representation of the propositional sentence with the input layer of agents as leaves and the output agent as the root.

    This may create 'redundant' nodes in hidden layers that would not exist in the minimal syntactic tree representation of a sentence with `input_size` atoms, when `input_size` is an odd number.

    Args:
        input_size: the size of the input layer of the network, e.g., the number of propositional atoms of the complex sentence to predict the truth value of.

    Returns:
        layers: a list of signaling network layers consisting of an input layer, an optional Sequential layer of hidden layers, and one output agent.
    """

    if input_size in [0, 1]:
        raise ValueError(
            f"A signaling network must consist of 2 or more agents. Received input size {input_size}"
        )

    # TODO: give each agent constructor kwargs so i can pass in the above args
    # default network

    input_layer = Layer([InputSender(input_size) for _ in range(input_size)])
    hidden_layers = []
    output_layer = OutputReceiver()

    layer_sizes = get_layer_sizes()
    # remove input and output layers
    layer_sizes = layer_sizes[1:-1]

    if layer_sizes:
        # populate each hidden layer with agents
        hidden_layers = Sequential(
            [Layer([HiddenSignaler(size) for _ in range(size)]) for size in layer_sizes]
        )

    layers = [
        layer
        for layer in [input_layer, hidden_layers, output_layer]
        if layer is not None
    ]
    return layers


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
        # x = list(example["input"])  # copy and shuffle order
        # random.shuffle(x)

        y = example["label"]
        y_hat = net(x)

        num_correct += 1 if y_hat == y else 0
        # net.update(reward_amount=0)

    net.train()
    return num_correct / num_rounds


def get_layer_sizes(input_size: int, topology: str = "binary-tree") -> list[int]:
    """Given an input size, construct a list containing the size of each layer of a network for mapping the input size to a singular output. By default constructs a binary tree graph topology.

    Args:
        input_size: the size of the input layer, corresponding to the length of a boolean formula in atoms, and the number of leaf nodes of the syntactic tree.

        topology: the network topology to construct. For mapping arbitary input sizes to a singular output, the shape must be a bottleneck, but in principle it may have a very long 'neck'.
    """
    if topology == "binary-tree":

        # create a list of the sizes of each hidden layer
        layer_sizes = list(
            reversed([2**j for j in range(0, int(np.ceil(np.log2(input_size))) + 1)])
        )
    else:
        raise ValueError(
            "Cannot support additional network topologies. Please construct a binary tree network."
        )

    return layer_sizes
