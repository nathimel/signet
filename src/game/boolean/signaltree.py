import numpy as np
from agents.module import (
    Sequential,
    Layer,
)
from agents.basic import (
    BooleanTranslator,
    Compressor,
)
from languages import Signal
from typing import Any

from game.boolean.functional import get_quaternary_receiver


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
        """Construct a signaling network for learning to predict the truth values of complex sentences of propositional logic.

        The idea is that the optimal network is a binary tree, corresponding to a the syntactic tree representation of the propositional sentence with the input layer of agents as leaves and the output agent as the root.
        """
        if input_size in [0, 1]:
            raise ValueError(
                f"A signaling network must consist of 2 or more agents. Received input size {input_size}"
            )

        # construct layers of tree network
        self.num_layers = input_size - 1
        self.num_hidden_layers = self.num_layers - 1
        self.hidden_layers = Sequential(
            layers=[
                HiddenSignalingLayer(
                    hidden_signalers=[
                        HiddenSignaler(  # each layer is a single agent...
                            input_size=input_size + i
                        )
                    ]
                )
                for i in range(self.num_hidden_layers)
            ]
        )
        self.output_layer = OutputReceiver(
            input_size=input_size + len(self.hidden_layers.layers)
        )
        super().__init__(layers=[self.hidden_layers, self.output_layer])
        print("hidden layers: ")
        print([agent for layer in self.hidden_layers.layers for agent in layer.agents])

    def forward(self, x: list[Signal]) -> Any:
        """A forward pass of the SignalTree.

        Args:
            x: a list of signals representing the truth values of a complex sentence of propositional logic

        Returns:
            a State (act) corresponding to the truth value of the complex sentence to learn."""
        return super().forward(x)


##############################################################################
# Component agents
##############################################################################


class HiddenSignaler(Sequential):
    """The basic building block of 'hidden' layers of a signaling network, consisting of a Compressor unit to get a composite (binary, e.g. "00") signal from the previous layer as input, and a BooleanTranslator to send a simple (unary, e.g. "0") signal as output."""

    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        # compressor uses two layers of attention to learn to compose two signals of input
        self.compressor = Compressor(input_size)

        # translator learns boolean function by mapping a space of four composite (boolean pair) signals to two simple boolean signals
        self.translator = BooleanTranslator()

        super().__init__(layers=[self.compressor, self.translator])

    def forward(self, x: list[Signal]) -> Any:
        return super().forward(x)

    def __str__(self) -> str:
        return f"HiddenSignaler of input size {self.input_size}"


class HiddenSignalingLayer(Layer):
    """A Hidden signaling 'layer' returns the output of hidden signaler outputs appended to the list of signals input to the layer."""

    def __init__(self, hidden_signalers: list[HiddenSignaler]) -> None:
        super().__init__(agents=hidden_signalers)

    def forward(self, x: list[Signal]) -> list[Signal]:
        signals = super().forward(x)
        return x + signals


class OutputReceiver(Sequential):
    """The final output agent for a signaling network is a module with a compressor unit to combine two signals, and a receiver to map this composite signal into an act."""

    def __init__(self, input_size: int) -> None:
        """By default input size is 2 for this 'root node' of a binary tree shaped network."""
        self.input_size = input_size
        self.compressor = Compressor(
            input_size
        )  # compressor works as in HiddenSignaler
        self.receiver = (
            get_quaternary_receiver()
        )  # receiver maps a space of four composite signals to two acts (representing truth values)
        super().__init__(layers=[self.compressor, self.receiver])

    def forward(self, x) -> Any:
        return super().forward(x)
