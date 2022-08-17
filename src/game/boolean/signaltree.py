from agents.module import (
    Sequential,
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
        self.layer_0 = HiddenSignaler(input_size)
        self.layer_1 = OutputReceiver(input_size + 1)
        super().__init__(layers=[self.layer_0, self.layer_1])

    def forward(self, x: list[Signal]) -> Any:
        """A forward pass of the SignalTree.

        Args:
            x: a list of signals representing the truth values of a complex sentence of propositional logic

        Returns:
            a State (act) corresponding to the truth value of the complex sentence to learn."""
        hidden_signal = self.layer_0(x)
        act = self.layer_1(x + [hidden_signal])
        return act


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
        # print("output receiver forward called.")
        return super().forward(x)
