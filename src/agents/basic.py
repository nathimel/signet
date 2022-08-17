import random
import numpy as np
from abc import abstractmethod
from altk.effcomm.agent import Speaker, Listener
from altk.language.semantics import Meaning
from languages import (
    Signal,
    SignalMeaning,
    SignalingLanguage,
    State,
    BooleanStateSpace,
)
from agents.module import (
    SignalingModule,
    Sequential,
)

from functools import reduce
from typing import Any, Union


# def compose_signals(signal_a: Signal, signal_b: Signal) -> Signal:
    # return Signal(f"{signal_a.form}{signal_b.form}")

# def compose_states(state_a: State, state_b: State) -> Signal:
    # return State(f"{state_a.name}{state_b.name}")

def compose(*x: list[Union[State, Signal]]) -> Union[State, Signal]:
    # just two states for now:
    """Map a pair of states or signals to their composition."""
    a, b = x
    if isinstance(a, State):
        return State(name=f"{a.name}{b.name}")
    if isinstance(b, Signal):
        return Signal(form=f"{a.form}{b.form}", meaning=BooleanStateSpace().referents)



##############################################################################
# Basic signaling agent wrappers for ALTK agents
##############################################################################


class Sender(Speaker):
    """A Sender agent in a signaling game chooses a signal given an observed state of nature, according to P(signal | state)."""

    def __init__(
        self,
        language: SignalingLanguage,
        weights=None,
        name: str = None,
    ):
        super().__init__(language, name=name)
        self.shape = (len(self.language.universe), len(self.language))
        self.initialize_weights(weights)

    def encode(self, state: Meaning) -> Signal:
        """Choose a signal given the state of nature observed, e.g. encode a discrete input as a discrete symbol."""
        index = self.sample_policy(index=self.referent_to_index(state))
        return self.index_to_expression(index)

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        """Map a policy to a `(state, signal)` index."""
        return (
            self.referent_to_index(policy["referent"]),
            self.expression_to_index(policy["expression"]),
        )


class Receiver(Listener):
    """A Receiver agent in a signaling game chooses an action=state given a signal they received, according to P(state | signal)."""

    def __init__(self, language: SignalingLanguage, weights=None, name: str = None):
        super().__init__(language, name=name)
        self.shape = (len(self.language), len(self.language.universe))
        self.initialize_weights(weights=weights)

    def decode(self, signal: Signal) -> SignalMeaning:
        """Choose an action given the signal received, e.g. decode a target state given its discrete encoding."""
        index = self.sample_policy(index=self.expression_to_index(signal))
        return self.index_to_referent(index)

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        """Map a policy to a `(signal, state)` index."""
        return (
            self.expression_to_index(policy["expression"]),
            self.referent_to_index(policy["referent"]),
        )


##############################################################################
# Actual signalers
##############################################################################


class SenderModule(SignalingModule):
    """Basic Sender wrapped in SignalingModule."""

    def __init__(self, sender: Sender) -> None:
        self.sender = sender
        super().__init__(parameters=self.sender.weights)

    def forward(self, x: State) -> Signal:
        signal = self.sender.encode(x)
        if self.train_mode:
            self.history.append({"referent": x, "expression": signal})
        return signal

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        return self.sender.policy_to_indices(policy)

    def to_language(self, **kwargs) -> SignalingLanguage:
        return self.sender.to_language(**kwargs)


class ReceiverModule(SignalingModule):
    """Basic Receiver wrapped in SignalingModule."""

    def __init__(self, receiver: Receiver) -> None:
        self.receiver = receiver
        super().__init__(parameters=self.receiver.weights)

    def forward(self, x: Signal) -> State:
        state = self.receiver.decode(signal=x)
        if self.train_mode:
            self.history.append({"referent": state, "expression": x})
        return state

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        return self.receiver.policy_to_indices(policy)

    def to_language(self, **kwargs) -> SignalingLanguage:
        return self.receiver.to_language(**kwargs)


# sanity check
class SSRReceiver(ReceiverModule):
    def forward(self, x: list[Signal]) -> State:
        return super().forward(x=compose(*x))


"""Composite modules only need to store their sub-agents, and call the forward and reward api of these sub-agents. These agents are assumed to be already instantiated, which means that their required and optional parameters should have been set, and now their internal parameters shouldn't be touched."""


class ReceiverSender(Sequential):
    def __init__(
        self,
        receiver: ReceiverModule,
        sender: SenderModule,
        name: str = None,
    ):
        """Construct a ReceiverSender by passing in a pair of languages.

        Args:

        """
        # initalize sub-agents
        self.receiver = receiver
        self.sender = sender
        self.name = name
        super().__init__(layers=[self.receiver, self.sender])


class AttentionAgent(SignalingModule):
    """An AttentionAgent implements a simple form of attention by sampling a single element of a many-element input (a list of signals or states). The probability of sampling a given element of the total input evolves under simple reinforcement learning."""

    def __init__(self, input_size: int):
        # Create a weight vector to represent the probability distribution over elements to pay attention to
        self.input_size = input_size
        super().__init__(parameters=np.ones(self.input_size))

    def forward(self, x: list[Any]) -> Any:
        # if not self.train_mode:
        # print("in test mode and sampling input with history: ")
        # print(self.history)
        return self.sample_input(x)

    def sample_input(self, x: list[Any]) -> Any:
        # sample an index
        index = np.random.choice(
            a=range(self.input_size),
            p=self.parameters / self.parameters.sum(),
        )
        output = x[index]

        if len(self.history) != 0:
            raise ValueError(
                f"The length of history before pushing a policy should be empty. Check that update() was called. The value of history: {self.history}"
            )
        if self.train_mode:
            # print("pushing a policy")
            self.history.append({"index": index})

        return output

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        # a singleton tuple
        return tuple([policy["index"]])

    def to_language(self, **kwargs) -> SignalingLanguage:
        """An attention agent is not communicative."""
        # yet
        return None


class AttentionSignaler(Sequential):
    """An AttentionSignaler module takes a complex input of signals or states, uses its attention layer to sample one of the elements, and passes the result to a signaler (either a SenderModule or ReceiverModule), and returns the resulting signaler's output."""

    def __init__(
        self,
        attention_layer: AttentionAgent,
        signaler: Union[SenderModule, ReceiverModule],
    ) -> None:
        self.attention_layer = attention_layer
        self.signaler = signaler
        super().__init__(layers=[self.attention_layer, self.signaler])


class Compressor(Sequential):
    """A Compressor module is a unit with two attention heads. It takes a list of signals as input, chooses two and combines them (as one of four possible combinations). This composite signal can become input to another agent.

    The Compressor is named so because it can compress an arbitrarily large input space to a single output, by sampling a pair of inputs and combining them.

    Attributes:
        attention_1: an AttentionAgent to sample the first input signal

        attention_2: an AttentionAgent to sample the second input signal
    """

    def __init__(
        self,
        input_size: int,
    ) -> None:
        """Construct a Compressor module.

        Args:
            input_size: the number of incoming signals [states] to compress to a single signal [state].
        """
        self.input_size = input_size
        self.attention_1 = AttentionAgent(self.input_size)  # shape (input_size, 1)
        self.attention_2 = AttentionAgent(self.input_size)  # shape (input_size, 1)
        super().__init__(layers=[self.attention_1, self.attention_2])

    def forward(self, x: list[Any]) -> Any:
        # print("input to compressor forward: ", x)
        # return compose_signals(
        # return compose_states(
        return compose(
            self.attention_1(x),
            self.attention_2(x),
        )


##############################################################################
# Baseline agents
##############################################################################


class Baseline(SignalingModule):
    """Baseline module that does not learn."""

    def update(self, reward_amount: float = 0) -> None:
        pass


class Bottom(Baseline):
    """Always returns 0."""

    def forward(self, x) -> State:
        return State(name="0")


class Top(Baseline):
    """Always returns 1."""

    def forward(self, x) -> State:
        return State(name="1")


class Random(Baseline):
    """Randomly return a state."""

    def forward(self, x) -> State:
        return random.choice([State(name="0"), State(name="1")])
