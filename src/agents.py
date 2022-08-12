from abc import abstractmethod
import numpy as np
from altk.effcomm.agent import CommunicativeAgent, Speaker, Listener
from altk.language.semantics import Meaning
from languages import Signal, SignalMeaning, SignalingLanguage, State, get_binary_language, get_quaternary_language
from typing import Any, Union

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
# Signaling Network Modules
##############################################################################


class SignalingModule:
    """The basic building block of a signaling network. Every agent in the network should inherit from this class.

    Attributes:
        parameters: a dict containing the learnable parameters for the agent.
        history: a stack to represent the (most recent) policies taken by the agent.
    """

    def __init__(self, parameters: np.ndarray = None) -> None:
        self.history = []
        self._parameters = parameters

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @parameters.setter
    def parameters(self, p: np.ndarray) -> None:
        self._parameters = p

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    @abstractmethod
    def forward(self, x) -> Any:
        raise NotImplementedError

    def update(self, reward_amount: float = 0) -> None:
        """Perform a learning update on the module, by optionally rewarding based on the last policy taken, and clearing the policy history for the next forward pass."""
        if reward_amount:
            self.reward(amount=reward_amount)
        self.reset_history()

    def reward(self, amount: float) -> None:

        if len(self.history) != 1:
            raise ValueError(
                f"Length of history must be exactly 1 to extract a unique policy to reward. Received history: {self.history}"
            )

        policy = self.history.pop()
        indices = self.policy_to_indices(policy)
        self.parameters[indices] += amount

    def reset_history(self) -> None:
        """Call this function after every forward pass through a module, whether it was rewarded or not."""
        self.history = []

    @abstractmethod
    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        """Returns a tuple containing one or more indices representing the location of the parameter for a policy."""
        raise NotImplementedError


class SenderModule(SignalingModule):
    """Basic Sender wrapped in SignalingModule."""

    def __init__(self, sender: Sender) -> None:
        self.sender = sender
        super().__init__(parameters=self.sender.weights)

    def forward(self, x: State) -> Signal:
        signal = self.sender.encode(x)
        self.history.append({"referent": x, "expression": signal})
        return signal

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        return self.sender.policy_to_indices(policy)


class ReceiverModule(SignalingModule):
    """Basic Receiver wrapped in SignalingModule."""

    def __init__(self, receiver: Receiver) -> None:
        self.receiver = receiver
        super().__init__(parameters=self.receiver.weights)

    def forward(self, x: Signal) -> State:
        state = self.receiver.decode(signal=x)
        self.history.append({"referent": state, "expression": x})
        return state

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        return self.receiver.policy_to_indices(policy)


"""Composite modules only need to store their sub-agents, and call the forward and reward api of these sub-agents. These agents are assumed to be already instantiated, which means that their required and optional parameters should have been set, and now their internal parameters shouldn't be touched."""


class ReceiverSender(SignalingModule):
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
        super().__init__()

    def forward(self, x: Signal) -> Signal:
        # no need to touch `history`
        return self.sender(self.receiver(x))

    def update(self, reward_amount: float = 0) -> None:
        self.receiver.update(reward_amount)
        self.sender.update(reward_amount)


class AttentionAgent(SignalingModule):
    """An AttentionAgent implements a simple form of attention by sampling a single element of a many-element input (a list of signals or states). The probability of sampling a given element of the total input evolves under simple reinforcement learning."""

    def __init__(self, input_size: int):
        # Create a weight vector to represent the probability distribution over elements to pay attention to
        self.input_size = input_size
        super().__init__(parameters=np.ones(self.input_size))

    def forward(self, x: list[Any]) -> Any:
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

        self.history.append({"index": index})

        return output

    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        # a singleton tuple
        return policy["index"]


class AttentionSignaler(SignalingModule):
    """An AttentionSignaler module takes a complex input of signals or states, uses its attention layer to sample one of the elements, and passes the result to a signaler (either a SenderModule or ReceiverModule), and returns the resulting signaler's output."""

    def __init__(
        self,
        attention_layer: AttentionAgent,
        signaler: Union[SenderModule, ReceiverModule],
    ) -> None:
        self.attention_layer = attention_layer
        self.signaler = signaler
        super().__init__()

    def forward(self, x: list[Any]) -> Any:
        return self.signaler(self.attention_layer(x))

    def update(self, reward_amount: float = 0) -> None:
        self.attention_layer.update(reward_amount)
        self.signaler.update(reward_amount)

class Compressor(SignalingModule):
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
            input_size: the number of incoming signals to compress to a single signal.
        """
        self.input_size = input_size
        self.attention_1 = AttentionAgent(self.input_size)  # shape (input_size, 1)
        self.attention_2 = AttentionAgent(self.input_size)  # shape (input_size, 1)
        super().__init__()

    def forward(self, x: list[Signal]) -> Signal:
        return compose(
            self.attention_1(x), 
            self.attention_2(x),
            )

    def update(self, reward_amount: float = 0) -> None:
        self.attention_1.update(reward_amount)
        self.attention_2.update(reward_amount)


##############################################################################
# Main game agents
##############################################################################

class InputSender(AttentionSignaler):
    def __init__(self, input_size: int) -> None:
        super().__init__(
            attention_layer=AttentionAgent(input_size),
            signaler=get_sender(),
        )

class HiddenSignaler(SignalingModule):
    """The basic building block of 'hidden' layers of a signaling network, consisting of a Compressor unit to get a composite (binary, e.g. "00") signal from the previous layer as input, and a ReceiverSender to send a simple (unary, e.g. "0") signal as output.
    """
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.compressor = Compressor(input_size)
        self.receiver_sender = get_receiver_sender()

    def forward(self, x) -> Any:
        return self.receiver_sender(self.compressor(x))

    def update(self, reward_amount: float = 0) -> None:
        self.compressor.update(reward_amount)
        self.receiver_sender.update(reward_amount)

class OutputReceiver(SignalingModule):

    """The final output agent for a signaling network is a module with a compressor unit to combine two signals, and a receiver to map this composite signal into an act.
    """
    def __init__(self, input_size: int = 2) -> None:
        """By default input size is 2 for this 'root node' of a binary tree shaped network."""
        self.input_size = input_size
        self.compressor = Compressor(input_size)
        self.receiver = get_quaternary_receiver()

    def forward(self, x) -> Any:
        return self.receiver(self.compressor(x))

    def update(self, reward_amount: float = 0) -> None:
        self.compressor.update(reward_amount)
        self.receiver.update(reward_amount)


##############################################################################
# Helper default functions 
# 
# for creating the agents typically used
# in predicting the truth values of boolean sentences with signaling networks
##############################################################################


def get_sender() -> SenderModule:
    """Get a 2 state, 2 signal SenderModule instance initialized for boolean games."""
    return SenderModule(sender=Sender(language=get_binary_language()))


def get_receiver() -> ReceiverModule:
    """Get a ReceiverModule instance initialized for boolean games."""
    return ReceiverModule(receiver=Receiver(language=get_binary_language()))


def get_quaternary_receiver() -> ReceiverModule:
    """Get a 4 signal, 2 state ReceiverModule instsance initialized for boolean games."""
    return ReceiverModule(receiver=Receiver(language=get_quaternary_language()))


def get_receiver_sender() -> ReceiverSender:
    """Get a ReceiverSender instance initialized for boolean games."""
    return ReceiverSender(
        receiver=get_quaternary_receiver(),
        sender=get_sender(),
    )

def compose(signal_a: Signal, signal_b: Signal) -> Signal:
    return Signal(f"{signal_a.form}{signal_b.form}")
