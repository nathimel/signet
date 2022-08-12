import game
import numpy as np
from agents import SignalingModule, AttentionSender, AttentionSignaler, Compressor
from languages import State, Signal
from functools import reduce
from typing import Any, Callable


class SignalTree(SignalingModule):
    """Signaling network for learning functions from lists of states (e.g., truth values of propositions) to states (e.g., the truth value of a complex proposition).

    A SignalTree is a signaling network with one agent as the final output classifier.

    For the task of predicting the truth values of complex propositional logic sentences from a list of their atoms' truth values, a natural network topology when agents are limited to binary input is the sentence's (binary) syntactic tree representation.
    """

    def __init__(self, input_size: int) -> None:
        # initialize layers
        self.layers = build_triangle_network(input_size)

        # wrap each layer of agents as a function
        self.input_layer = lambda x: [agent(x) for agent in self.layers["input"]]

        if self.layers["hidden"]:
            self.hidden_layers = lambda x: sequential(self.layers["hidden"], x)

        self.output_layer = lambda x: self.layers["output"][0](x)

    def forward(self, x: list[State]) -> State:
        x_hat = self.input_layer(x)
        if self.layers["hidden"]:
            x_hat = self.hidden_layers(x_hat)
        output = self.output_layer(x_hat)
        return output

    def reward(self, amount: float) -> None:
        [
            agent.reward(amount)
            for layer in [self.layers["input"]] + self.layers["hidden"] + [self.layers["output"]] for agent in layer
        ]
    
    def punish(self, amount: float) -> None:
        [
            agent.punish(amount)
            for layer in [self.layers["input"]] + self.layers["hidden"] + [self.layers["output"]] for agent in layer
        ]        


def sequential(
    layers: list[list[SignalingModule]], input_signals: list[Signal]
) -> Callable[[list[Any]], list[Any]]:
    """Converts a list of layers to a function reducing the result of each applied in order."""
    funcs = [lambda x: [agent(x) for agent in layer] for layer in layers]
    result = reduce(lambda res, f: f(res), funcs, input_signals)
    return result


def build_triangle_network(input_size: int = 2) -> dict[str, list[SignalingModule]]:
    """Construct a signaling network to have a triangle shape, so that each layer doubles in size moving backwards, starting from the single output agent.

    The idea is that the optimal network is a binary tree, corresponding to a the syntactic tree representation of the propositional sentence with the input layer of agents as leaves and the output agent as the root.

    This may create 'redundant' nodes in hidden layers that would not exist in the minimal syntactic tree representation of a sentence with `input_size` atoms, when `input_size` is an odd number.

    Args:
        input_size: the size of the input layer of the network, e.g., the number of propositional atoms of the complex sentence to predict the truth value of.

    Returns:

        layers: a dict containing the levels of the binary tree, corresponding to layers of the signaling network. Of the form
            {
                "input": (list of agents),
                "hidden": (list of lists of agents),
                "output": (one agent),
            }
    """

    if input_size in [0, 1]:
        raise ValueError(
            f"A signaling network must consist of 2 or more agents. Received input size {input_size}"
        )

    # default network
    layers = {
        "input": [AttentionSender(input_size) for _ in range(input_size)],
        "hidden": [],
        "output": [game.get_output_agent()],
    }

    # create a list of the sizes of each hidden layer
    layer_sizes = list(
        reversed([2**j for j in range(0, int(np.ceil(np.log2(input_size))) + 1)])
    )
    # remove input and output layers
    layer_sizes.pop(-1)
    layer_sizes.pop(0)

    if layer_sizes:
        # populate each hidden layer with agents
        layers["hidden"] = [
            [game.get_compressor(size) for _ in range(size)] for size in layer_sizes
        ]

    return layers
