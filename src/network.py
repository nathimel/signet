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
        self.agents = build_triangle_network(input_size)

        # wrap each layer of agents as a function
        self.input_layer = Layer(self.agents["input"])
        self.hidden_layers = None
        if self.agents["hidden"]:
            self.hidden_layers = Sequential(
                Layer(layer) for layer in self.agents["hidden"]
            )
        (self.output_layer,) = self.agents["output"]

        self.all_layers = Sequential(
            [
                layer
                for layer in [self.input_layer, self.hidden_layers, self.output_layer]
                if layer is not None
            ]
        )

        super().__init__()

    def forward(self, x: list[State]) -> State:
        x_hat = self.input_layer(x)
        if self.agents["hidden"]:
            x_hat = self.hidden_layers(x_hat)
        return self.output_layer(x_hat)

    def update(self, reward_amount: float = 0) -> None:
        self.input_layer.update(reward_amount)
        self.output_layer.update(reward_amount)


class Layer(SignalingModule):
    def __init__(self, agents: list[SignalingModule]) -> None:
        self.agents = agents

    def forward(self, x) -> Any:
        return [agent(x) for agent in self.agents]

    def update(self, reward_amount: float = 0) -> None:
        [agent.update(reward_amount) for agent in self.agents]


class Sequential(SignalingModule):
    """Constructs a module consisting of the result of a list of layers applied to each other in order using `reduce`."""

    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def forward(self, x) -> Any:
        return reduce(
            lambda res, f: f(res),
            [lambda x_: layer(x_) for layer in self.layers],
            x,
        )

    def update(self, reward_amount: float = 0) -> None:
        [layer.update(reward_amount) for layer in self.layers]


def build_triangle_network(input_size: int = 2) -> dict[str, list[SignalingModule]]:
    """Construct a signaling network to have a triangle shape, so that each layer doubles in size moving backwards, starting from the single output agent.

    The idea is that the optimal network is a binary tree, corresponding to a the syntactic tree representation of the propositional sentence with the input layer of agents as leaves and the output agent as the root.

    This may create 'redundant' nodes in hidden layers that would not exist in the minimal syntactic tree representation of a sentence with `input_size` atoms, when `input_size` is an odd number.

    Args:
        input_size: the size of the input layer of the network, e.g., the number of propositional atoms of the complex sentence to predict the truth value of.

    Returns:

        agents: a dict containing the levels of the binary tree, corresponding to layers of the signaling network. Of the form
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
    agents = {
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
        agents["hidden"] = [
            [game.get_compressor(size) for _ in range(size)] for size in layer_sizes
        ]

    return agents
