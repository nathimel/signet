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
    get_binary_language,
    get_four_state_two_signal_language,
    get_two_state_four_signal_language,
)
from functools import reduce
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

    def __init__(
        self,
        parameters: np.ndarray = None,
        learner: str = "Roth-Erev",
        learning_rate: float = 1.0,
        name: str = None,
    ) -> None:
        """The base constructor for a signaling module used to build signaling networks.

        Args:
            parameters: optional numpy array of weights for an agent.

            learner: {"Roth-Erev", "Bush-Mosteller"} the kind of learning dynamics to implement. Default is "Roth-Erev".

            learning_rate: a float determining speed of learning. If Roth-Erev learning, multiply rewards by this value. Bush-Mosteller rewards requires a learning_rate in the interval [0,1].
        """
        self.history = []
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.name = name

        # In training mode or not / whether to update parameters
        self.train_mode = True
        self.frozen = False

        if learner == "Roth-Erev":
            # assign reward func
            self.reward_func = roth_erev_reinforce
        elif learner == "Bush-Mosteller":
            # check learning rate
            if self.learning_rate < 0.0 or self.learning_rate > 1.0:
                raise ValueError(
                    f"learning rate must be in [0,1] for Bush-Mosteller learning. Received: {learning_rate}."
                )
            # assign reward func
            self.reward_func = bush_mosteller_reinforce
            # normalize vectors to sum to 1.0
            if self.train_mode:
                if self.parameters is not None:
                    axis = self.parameters.ndim - 1
                    self.parameters /= self.parameters.sum(axis=axis, keepdims=True)

        else:
            raise ValueError(
                f"The argument `learner` must be either 'Roth-erev' or 'Bush-Mosteller'. Received: {learner}"
            )

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

    def update(self, reward_amount: float) -> None:
        """Perform a learning update on the module, by optionally rewarding based on the last policy taken, and clearing the policy history for the next forward pass."""
        if self.train_mode:
            self.reward(amount=self.learning_rate * reward_amount)

    def reward(self, amount: float) -> None:

        if len(self.history) != 1:
            raise ValueError(
                f"Length of history must be exactly 1 to extract a unique policy to reward. Received history: {self.history}"
            )

        policy = self.history.pop()
        indices = self.policy_to_indices(policy)
        if not self.frozen:
            self.parameters = self.reward_func(
                self.parameters, indices, amount, learning_rate=self.learning_rate
            )

    def reset_history(self) -> None:
        """Call this function after every forward pass through a module, whether it was rewarded or not."""
        self.history = []

    @abstractmethod
    def policy_to_indices(self, policy: dict[str, Any]) -> tuple[int]:
        """Returns a tuple containing one or more indices representing the location of the parameter for a policy."""
        raise NotImplementedError

    def train(self) -> None:
        """Set module to training mode."""
        # if self.frozen:
        # raise Exception("Parameters frozen but tried to train.")
        self.train_mode = True

    def test(self) -> None:
        """Set module to testing mode."""
        self.train_mode = False

    def freeze(self) -> None:
        """Freeze parameters to not be updated by blocking `train`."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze parameters to be updated by `train`."""
        self.frozen = False

    def to_language(self, **kwargs) -> SignalingLanguage:
        """Return the learned 'language', e.g. pairing of signals and states, for an agent, if they are a communicative agent."""
        raise NotImplementedError


class Layer(SignalingModule):
    """A layer is a list of agents that can act as one composite agent."""

    def __init__(self, agents: list[SignalingModule]) -> None:
        self.agents = agents

    def forward(self, x) -> Any:
        return [agent(x) for agent in self.agents]

    def update(self, reward_amount: float) -> None:
        [agent.update(reward_amount) for agent in self.agents]

    def train(self) -> None:
        [agent.train() for agent in self.agents]

    def test(self) -> None:
        [agent.test() for agent in self.agents]

    def freeze(self) -> None:
        [agent.freeze() for agent in self.agents]

    def unfreeze(self) -> None:
        [agent.unfreeze() for agent in self.agents]

    def to_language(self, **kwargs) -> SignalingLanguage:
        return [agent.to_language(**kwargs) for agent in self.agents]


class Sequential(SignalingModule):
    """Constructs a module consisting of the result of a list of layers applied to each other in order using `reduce`."""

    def __init__(self, layers: list[Layer]) -> None:
        """Take a list of layers of agents to compose."""
        self.layers = layers

    def forward(self, x) -> Any:
        return reduce(
            lambda res, f: f(res),
            [layer for layer in self.layers],
            x,
        )

    def update(self, reward_amount: float) -> None:
        [layer.update(reward_amount) for layer in self.layers]

    def train(self) -> None:
        [layer.train() for layer in self.layers]

    def test(self) -> None:
        [layer.test() for layer in self.layers]

    def freeze(self) -> None:
        [layer.freeze() for layer in self.layers]

    def unfreeze(self) -> None:
        [layer.unfreeze() for layer in self.layers]

    def to_language(self, **kwargs) -> SignalingLanguage:
        return [layer.to_language(**kwargs) for layer in self.layers]


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
            input_size: the number of incoming signals to compress to a single signal.
        """
        self.input_size = input_size
        self.attention_1 = AttentionAgent(self.input_size)  # shape (input_size, 1)
        self.attention_2 = AttentionAgent(self.input_size)  # shape (input_size, 1)
        super().__init__(layers=[self.attention_1, self.attention_2])

    def forward(self, x: list[Signal]) -> Signal:
        # print("input to compressor forward: ", x)
        return compose(
            self.attention_1(x),
            self.attention_2(x),
        )


##############################################################################
# Main game agents
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
    """Get a 4 signal, 2 state ReceiverModule instance initialized for boolean games."""
    return ReceiverModule(
        receiver=Receiver(language=get_two_state_four_signal_language())
    )

def get_ssr_receiver() -> SSRReceiver:
    return SSRReceiver(receiver=Receiver(language=get_two_state_four_signal_language()))

def get_quaternary_sender() -> ReceiverModule:
    """Get a 4 state, 2 signal ReceiverModule instance initialized for boolean games."""
    return SenderModule(sender=Sender(language=get_four_state_two_signal_language()))



def get_receiver_sender() -> ReceiverSender:
    """Get a ReceiverSender instance initialized for boolean games."""
    return ReceiverSender(
        receiver=get_quaternary_receiver(),
        sender=get_sender(),
    )


def compose(signal_a: Signal, signal_b: Signal) -> Signal:
    return Signal(f"{signal_a.form}{signal_b.form}")


def bush_mosteller_reinforce(
    parameters: np.ndarray, indices: tuple[int], amount: float, **kwargs
) -> np.ndarray:
    """Update on policy A according to:
    prob_new(A) = prob_old(A) + \alpha(1 - prob_old(A))
    """
    vector_idx = indices[:-1]  # (possibly empty) first dimension
    act_idx = indices[-1]  #  last dimension to index within the act vector

    act_vector = parameters[vector_idx]

    if act_vector.sum() != 1.0:
        raise ValueError(
            f"The act vector must sum to 1.0 to represent a probability distribution. Value of act vector: {act_vector}. Value of all parameters: {parameters}."
        )

    if "learning_rate" not in kwargs:
        raise ValueError(
            "Bush-Mosteller reinforcement learning requires a learning rate, but none was passed."
        )
    prob_old_A = parameters[indices]

    delta = amount * (1 - prob_old_A)
    prob_new_A = prob_old_A + delta

    # renormalize remaining parameters until vector sums to one
    # by decrementing by delta / number of remaining parameters
    act_vector_new = act_vector - (delta / (len(act_vector) - 1))
    act_vector_new[act_idx] = prob_new_A

    if act_vector_new.sum() != 1.0:
        raise ValueError(
            f"Result of Bush-Mosteller update must result in a policy vector that sums to 1.0. Result was: {act_vector_new}"
        )

    parameters[vector_idx] = act_vector_new

    return parameters


def roth_erev_reinforce(
    parameters: np.ndarray,
    indices: tuple[int],
    amount: float,
    **kwargs,
) -> np.ndarray:
    """Increment the weight (accumulated rewards) of a policy specifed by `indices` by `amount`."""
    parameters[indices] += amount
    return parameters

def win_stay_lose_shift_inertia(
    parameters: np.ndarray,
    indices: tuple[int],
    amount: float,
    **kwargs,
)-> np.ndarray:
    """Win-stay lose-shift learning with inertia.
    
    Requires a history to determine the number of losses, and to switch if greater than $i$, the inertia.
    """
