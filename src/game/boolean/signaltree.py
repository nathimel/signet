import random
import numpy as np
from agents.module import (
    Layer,
    Sequential,
)
from agents.basic import (
    AttentionAgent,
    AttentionSignaler,
    Compressor,
)
from languages import State, Signal
from typing import Any

from game.boolean.functional import (
    get_quaternary_sender,
    get_ssr_receiver,
    get_sender,
    get_receiver_sender,
    get_quaternary_receiver,
    get_layer_sizes,
)


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


##############################################################################
# Component agents
##############################################################################


class InputSender(AttentionSignaler):
    def __init__(self, input_size: int) -> None:
        super().__init__(
            attention_layer=AttentionAgent(input_size),
            signaler=get_sender(),
        )

    def forward(self, x: list[State]) -> Signal:
        # print("Input sender forward called.")
        return super().forward(x)


class HiddenSignaler(Sequential):
    """The basic building block of 'hidden' layers of a signaling network, consisting of a Compressor unit to get a composite (binary, e.g. "00") signal from the previous layer as input, and a ReceiverSender to send a simple (unary, e.g. "0") signal as output."""

    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.compressor = Compressor(input_size)
        self.receiver_sender = get_receiver_sender()
        super().__init__(layers=[self.compressor, self.receiver_sender])

    def forward(self, x) -> Any:
        # print("hidden signaler forward called")
        return super().forward(x)


class OutputReceiver(Sequential):

    """The final output agent for a signaling network is a module with a compressor unit to combine two signals, and a receiver to map this composite signal into an act."""

    def __init__(self, input_size: int = 2) -> None:
        """By default input size is 2 for this 'root node' of a binary tree shaped network."""
        self.input_size = input_size
        self.compressor = Compressor(input_size)
        self.receiver = get_quaternary_receiver()
        super().__init__(layers=[self.compressor, self.receiver])

    def forward(self, x) -> Any:
        # print("output receiver forward called.")
        return super().forward(x)


####


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
    net = Sequential(layers=[sender_layer, receiver])

    return net
