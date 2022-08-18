import random
from tkinter import E
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
from typing import Any, Iterable, Union
from game.reinforce import construct_policy_maps, update_map


def compose(*x: Iterable[Union[State, Signal]]) -> Union[State, Signal]:
    """Map a list of states or signals to their composition."""
    if all(isinstance(item, State) for item in x):
        return State(name=f"{''.join([item.name for item in x])}")
    if all(isinstance(item, Signal) for item in x):
        return Signal(
            form=f"{''.join([item.form for item in x])}",
            meaning=BooleanStateSpace().referents,
        )


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

    def __init__(self, sender: Sender, **kwargs) -> None:
        self.sender = sender
        kwargs["parameters"] = self.sender.weights

        super().__init__(**kwargs)

        self.maps = construct_policy_maps(
            inputs=self.sender.language.universe.referents,
            outputs=self.sender.language.expressions,
        )
        self.map = update_map(self.maps)

    def forward(self, x: State) -> Signal:
        if self.learner == "WSLSwI":
            signal = self.wslswi_map(x)
        else:
            signal = self.sender.encode(x)
        if self.train_mode:
            self.push_policy({"referent": x, "expression": signal})
        return signal

    def wslswi_map(self, x: State) -> Signal:
        """Deterministically map a state to a signal if the map is not above WSlSwI inertia."""
        super().wslswi_map(x)

    def stay_or_shift(self, success: bool = True) -> None:
        policy = self.pop_policy()
        if not self.frozen:
            # increment the number of losses
            self.map[policy["referent"]]["losses"] += int(not success)
            # update policy if necessary
            if self.map[policy["referent"]]["losses"] > self.inertia:
                self.map = update_map(self.maps, self.map, inertia=self.inertia)

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        return self.sender.policy_to_indices(policy)

    def to_language(self, **kwargs) -> SignalingLanguage:
        return self.sender.to_language(**kwargs)


class ReceiverModule(SignalingModule):
    """Basic Receiver wrapped in SignalingModule."""

    def __init__(self, receiver: Receiver, **kwargs) -> None:
        self.receiver = receiver
        kwargs["parameters"] = self.receiver.weights
        super().__init__(**kwargs)

        self.maps = construct_policy_maps(
            inputs=self.receiver.language.expressions,
            outputs=self.receiver.language.universe.referents,
        )
        self.map = update_map(self.maps)

    def forward(self, x: Signal) -> State:
        if self.learner == "WSLSwI":
            state = self.wslswi_map(x)
        else:
            state = self.receiver.decode(signal=x)
        if self.train_mode:
            self.push_policy({"referent": state, "expression": x})

        return state

    def wslswi_map(self, x: Signal) -> State:
        """Deterministically map a state to a signal if the map is not above WSlSwI inertia."""
        return super().wslswi_map(x)

    def stay_or_shift(self, success: bool = True) -> None:
        policy = self.pop_policy()
        if not self.frozen:
            # increment the number of losses
            self.map[policy["expression"]]["losses"] += int(not success)
            # update policy if necessary
            if self.map[policy["expression"]]["losses"] > self.inertia:
                self.map = update_map(self.maps, self.map, inertia=self.inertia)

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        return self.receiver.policy_to_indices(policy)

    def to_language(self, **kwargs) -> SignalingLanguage:
        return self.receiver.to_language(**kwargs)


"""Composite modules only need to store their sub-agents, and call the forward and reward api of these sub-agents. These agents are assumed to be already instantiated, which means that their required and optional parameters should have been set, and now their internal parameters shouldn't be touched."""


class AttentionAgent(SignalingModule):
    """An AttentionAgent implements a simple form of attention by sampling a single element of a many-element input (a list of signals or states). The probability of sampling a given element of the total input evolves under simple reinforcement learning."""

    def __init__(self, input_size: int):
        # Create a weight vector to represent the probability distribution over elements to pay attention to
        self.input_size = input_size
        super().__init__(parameters=np.ones(self.input_size))

        self.maps = construct_policy_maps(
            inputs=[0],  # dummy
            outputs=range(input_size),
        )
        self.map = update_map(self.maps)

    def forward(self, x: list[Any]) -> Any:
        if self.learner == "WSLSwI":
            result = self.wslswi_map(x)
        else:
            result = self.sample_input(x)
        if self.train_mode:
            self.push_policy({"index": result["index"]})
        return result["output"]

    def sample_input(self, x: list[Any]) -> dict[str, Any]:
        # sample an index
        index = np.random.choice(
            a=range(self.input_size),
            p=self.parameters / self.parameters.sum(),
        )
        output = x[index]
        return {"index": index, "output": output}

    def wslswi_map(self, x) -> Any:
        """Note that attention does not rely on the value of input, only the size."""
        index = self.map[0]["output"]
        output = x[index]
        return {"index": index, "output": output}

    def stay_or_shift(self, success: bool = True) -> None:
        policy = self.pop_policy()
        if not self.frozen:
            # increment the number of losses
            self.map[0]["losses"] += int(not success) # recall attention does not depend on input, so we dummy index the input with 0
            # update policy if necessary
            if self.map[0]["losses"] > self.inertia:
                self.map = update_map(self.maps, self.map, inertia=self.inertia)

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        # a singleton tuple
        return tuple([policy["index"]])

    def to_language(self, **kwargs) -> SignalingLanguage:
        """Dummy function for now since attention is not communicative."""
        pass


class Translator(SignalingModule):
    """Maps a set of signals to a (possibly different sized) set of signals."""

    def __init__(
        self, source_signals: list[Signal], target_signals: list[Signal], **kwargs
    ) -> None:

        self.source_to_index = {
            signal: index for index, signal in enumerate(source_signals)
        }
        self.target_to_index = {
            signal: index for index, signal in enumerate(target_signals)
        }
        self.index_to_source_signal = tuple(source_signals)
        self.index_to_target_signal = tuple(target_signals)
        translation_weights = np.ones((len(source_signals), len(target_signals)))

        kwargs["parameters"] = translation_weights
        super().__init__(**kwargs)

        self.maps = construct_policy_maps(
            inputs=source_signals,
            outputs=target_signals,
        )
        self.map = update_map(self.maps)

    def forward(self, x: Signal) -> Any:
        if self.learner == "WSLSwI":
            signal = self.wslswi_map(x)
        else:
            signal = self.translate(x)
        if self.train_mode:
            self.push_policy({"source": x, "target": signal})
        return signal

    def translate(self, source: Signal) -> Signal:
        """Map a source signal to a target signal."""
        index = self.sample_policy(index=self.source_to_index[source])
        target = self.index_to_target_signal[index]
        return target

    def sample_policy(self, index: int) -> int:
        """Sample a communicative policy by uniformly sampling from a row vector of the agent's weight matrix specified by the index.

        Args:
            index: the integer index representing a row of the weight matrix.

        Returns:
            the integer index of the agent's choice
        """
        choices = self.parameters[index]
        choices_normalized = choices / choices.sum()
        return np.random.choice(
            a=range(len(choices)),
            p=choices_normalized,
        )

    def wslswi_map(self, x: Signal) -> Signal:
        return super().wslswi_map(x)

    def stay_or_shift(self, success: bool = True) -> None:
        policy = self.pop_policy()
        if not self.frozen:
            # increment the number of losses
            self.map[policy["source"]]["losses"] += int(not success)
            # update policy if necessary
            if self.map[policy["source"]]["losses"] > self.inertia:
                self.map = update_map(self.maps, self.map, inertia=self.inertia)

    def policy_to_indices(self, policy: dict[str, State]) -> tuple[int]:
        """Maps a pair of (composite state, simple state) to numerical indices specifying the weight for updating this policy."""
        return (
            self.source_to_index[policy["source"]],
            self.target_to_index[policy["target"]],
        )


class BooleanTranslator(Translator):
    def __init__(self, **kwargs) -> None:
        source_signals = [
            Signal(form="00"),
            Signal(form="01"),
            Signal(form="10"),
            Signal(form="11"),
        ]
        target_signals = [
            Signal(form="0"),
            Signal(form="1"),
        ]
        super().__init__(source_signals, target_signals, **kwargs)


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
        composed = compose(
            self.attention_1(x),
            self.attention_2(x),
        )
        return composed


##############################################################################
# Baseline agents
##############################################################################


class Baseline(SignalingModule):
    """Baseline module that does not learn."""


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
