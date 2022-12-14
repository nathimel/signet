"""The main module abstractions used for building signaling networks."""
from abc import abstractmethod
import numpy as np
from functools import reduce
from languages import SignalingLanguage
from game.reinforce import roth_erev_reinforce, bush_mosteller_reinforce
from typing import Any, Callable

##############################################################################
# Signaling Network Modules
##############################################################################


class SignalingModule:
    """The basic building block of a signaling network. Every agent in the network should inherit from this class.

    Attributes:

        history: a stack to represent the list of policies taken by an agent.

        parameters: (optional) numpy array of weights for an agent.

        learner: (optional) {"Roth-Erev", "Bush-Mosteller"} the kind of learning dynamics to implement. Default is "Roth-Erev".

        learning_rate: (optional) a float determining speed of learning. If Roth-Erev learning, multiply rewards by this value. Bush-Mosteller rewards requires a learning_rate in the interval [0,1].
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """The base constructor for a signaling module used to build signaling networks."""

        parameters = None
        learner = "WSLSwI"
        learning_rate = 1.0
        inertia = 1
        name = None
        if "parameters" in kwargs:
            parameters = kwargs["parameters"]

        if "learner" in kwargs:
            learner = kwargs["learner"]

        if "learning_rate" in kwargs:
            learning_rate = kwargs["learning_rate"]

        if "name" in kwargs:
            name = kwargs["name"]

        if "inertia" in kwargs:
            inertia = kwargs["inertia"]

        self.name = name
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.history = []

        # In training mode or not / whether to update parameters
        self.train_mode = True
        self.frozen = False

        if learner == "Roth-Erev":
            self.learner = learner
            # assign reward func
            self.reward_func = roth_erev_reinforce
        elif learner == "Bush-Mosteller":
            # check learning rate
            if self.learning_rate < 0.0 or self.learning_rate > 1.0:
                raise ValueError(
                    f"learning rate must be in [0,1] for Bush-Mosteller learning. Received: {learning_rate}."
                )
            self.learner = learner
            # assign reward func
            self.reward_func = bush_mosteller_reinforce
            # normalize vectors to sum to 1.0
            if self.train_mode:
                if self.parameters is not None:
                    self.normalize_params()

        elif learner == "WSLSwI":
            self.learner = learner
            self.inertia = inertia
            self.maps = None
            self.map = None
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

    def update(
        self,
        reward_amount: float,
    ) -> None:
        """Perform a learning update on the module, by optionally rewarding based on the last policy taken, and clearing the policy history for the next forward pass.

        Args:
            reward_amount: the amount to update (scaled by learning rate, etc.) parameters by, if using reinforcement learning. If using Win Stay Lose Shift with Inertia, update parameters by incrementing losses if reward amount is 0.
        """
        if self.train_mode:
            if self.learner == "WSLSwI":
                # update via win stay lose shift with inertia
                if not self.frozen:
                    # adjust losses and shift if necessary
                    self.stay_or_shift(bool(reward_amount))
            else:
                # reinforcement
                self.reward(amount=self.learning_rate * reward_amount)

    def wslswi_map(self, x) -> Any:
        return self.map[x]["output"]

    def stay_or_shift(self, success: bool = True) -> None:
        """For now, the count is based on number of successes and failures, but we could generalize to a weighted score of success and failure, so that inertia represents a general threshold, instead of a minimum count of failures."""
        raise NotImplementedError

    def reward(self, amount: float) -> None:
        if not self.frozen:
            indices = self.policy_to_indices(self.pop_policy())
            self.parameters = self.reward_func(
                self.parameters, indices, amount, learning_rate=self.learning_rate
            )

    def reset_history(self) -> None:
        """Call this function after every forward pass through a module, whether it was rewarded or not."""
        self.history = []

    def pop_policy(self) -> tuple[int]:
        """Pop the most recent policy off the history stack, and return the corresponding indices."""
        if len(self.history) != 1:
            raise Exception(
                f"Length of history must be exactly 1 to extract a unique policy to reward. Received history: {self.history}"
            )
        policy = self.history.pop()
        return policy

    def push_policy(self, policy: dict[str, Any]) -> None:
        """Push a policy resulting from the forward pass onto the history stack."""
        if len(self.history) != 0:
            raise Exception(
                f"The length of history before pushing a policy should be empty. Check that update() was called. The value of history: {self.history}"
            )
        self.history.append(policy)

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

    # optimization and learning

    def normalize_params(self) -> None:
        """Normalize each row vector in parameters to sum to 1.0"""
        axis = self.parameters.ndim - 1
        self.parameters /= self.parameters.sum(axis=axis, keepdims=True)

    def rescale_params(self, alpha: float = 20) -> None:
        """Rescale the weights/parameters to have a maximum of `alpha`."""
        self.normalize_params()
        self.parameters *= alpha


class Layer:
    """A layer is a list of agents that can act as one composite agent."""

    def __init__(self, agents: list[SignalingModule]) -> None:
        self.agents = agents

    # def apply_to_agents(self, f: Callable, x = None):
    #     return [getattr(__o = agent, name = str(f))(x) for agent in self.agents]

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

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

    def rescale_params(self, alpha: float = 20) -> None:
        [agent.rescale_params(alpha) for agent in self.agents]

    def to_language(self, **kwargs) -> SignalingLanguage:
        return [agent.to_language(**kwargs) for agent in self.agents]


class Sequential:
    """Constructs a module consisting of the result of a list of layers applied to each other in order using `reduce`."""

    def __init__(self, layers: list[Layer]) -> None:
        """Take a list of layers of agents to compose."""
        self.layers = layers

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

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

    def rescale_params(self, alpha: float = 20) -> None:
        [layer.rescale_params(alpha) for layer in self.layers]

    def to_language(self, **kwargs) -> SignalingLanguage:
        return [layer.to_language(**kwargs) for layer in self.layers]
