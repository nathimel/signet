"""Functions for data generating used for boolean games."""
import numpy as np
from itertools import product
from typing import Any, Callable, Type, Union
from functools import reduce

from agents.basic import (
    Receiver,
    ReceiverModule,
    Sender,
    SenderModule,
)
from languages import (
    State,
    Signal,
    get_binary_language,
    get_four_state_two_signal_language,
    get_two_state_four_signal_language,
)

##############################################################################
# Data
##############################################################################

# Connectives


def AND(x: bool, y: bool) -> bool:
    return x and y


def OR(x: bool, y: bool) -> bool:
    return x or y


def XOR(x: bool, y: bool) -> bool:
    return x != y


def NAND(x: bool, y: bool) -> bool:
    return not (x and y)


def IMPLIES(x: bool, y: bool) -> bool:
    return not x or y


def IFF(x: bool, y: bool) -> bool:
    return x == y


def bool_to_type(x: bool, type: Type = State) -> Union[State, Signal]:
    return type(str(int(x)))


def n_ary_data(
    n: int,
    connective: Callable[[bool, bool], bool] = lambda x, y: x and y,
    input_type: Type = State,
    output_type: Type = State,
) -> list[dict[str, Any]]:
    """Constructs a dataset for a sentence of propositional logic as positive (states, act) examples for a signaling network to learn from.

    For now, we assume sentences to be iteratively composed of only one truth-connective.

    Args:
        n: the number of atomic propositions in the sentence

        connective: the connective to iteratively build the sentence.

        input_type: the types that the input list should consist of, either State or Signal

        output_type the type that the output label should be, either State or Signal.

    Returns:
        a list of examples, each of the form {"input": (...), "label": (...)}
    """
    f = lambda inputs: reduce(connective, inputs)
    examples = []
    assignments = list(
        product([False, True], repeat=n)
    )  # get all possible combinations of truth values
    for inputs in assignments:
        example = {
            "input": [
                bool_to_type(atom, input_type) for atom in inputs
            ],  # a list of states
            "label": bool_to_type(f(inputs), output_type),  # a state
        }
        examples.append(example)
    return examples


def numerical_data(
    n: int,
    connective: Callable[[bool, bool], bool] = lambda x, y: x and y,
) -> tuple[np.ndarray]:
    """
    Constructs a dataset for a sentence of propositional logic as positive (states, act) examples for a signaling network to learn from.

    For now, we assume sentences to be iteratively composed of only one truth-connective.

    Args:
        n: the number of atomic propositions in the sentence

        connective: the connective to iteratively build the sentence.

    Returns:
        (X, y) a tuple of numpy arrays of floats representing the dataset to train a learner on.
    """
    X = []
    y = []

    f = lambda inputs: reduce(connective, inputs)
    assignments = list(
        product([False, True], repeat=n)
    )  # get all possible combinations of truth values
    for inputs in assignments:
        x = np.array([float(item) for item in inputs])
        label = float(f(inputs))
        X.append(x)
        y.append(label)

    return (np.array(X), np.array(y))


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


def get_quaternary_sender() -> ReceiverModule:
    """Get a 4 state, 2 signal ReceiverModule instance initialized for boolean games."""
    return SenderModule(sender=Sender(language=get_four_state_two_signal_language()))


def get_layer_sizes(input_size: int, topology: str = "binary-tree") -> list[int]:
    """Given an input size, construct a list containing the size of each layer of a network for mapping the input size to a singular output. By default constructs a binary tree graph topology.

    Args:
        input_size: the size of the input layer, corresponding to the length of a boolean formula in atoms, and the number of leaf nodes of the syntactic tree.

        topology: the network topology to construct. For mapping arbitary input sizes to a singular output, the shape must be a bottleneck, but in principle it may have a very long 'neck'.
    """
    if topology == "binary-tree":

        # create a list of the sizes of each hidden layer
        layer_sizes = list(
            reversed([2**j for j in range(0, int(np.ceil(np.log2(input_size))) + 1)])
        )
    else:
        raise ValueError(
            "Cannot support additional network topologies. Please construct a binary tree network."
        )

    return layer_sizes
