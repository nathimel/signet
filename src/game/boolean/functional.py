"""Functions for data generating used for boolean games."""
import numpy as np
from languages import State
from itertools import product
from typing import Callable
from functools import reduce

from agents.basic import (
    Receiver,
    ReceiverModule,
    ReceiverSender,
    SSRReceiver,
    Sender,
    SenderModule,
)
from languages import (
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


# Utility functions


def bool_to_state(x: bool) -> State:
    return State(str(int(x)))


def generate_data(sentence: str = "p and q") -> list[dict[str, State]]:
    """Given a boolean function, generate a dataset of positive examples to train agents on corresponding to the table representation of the function.

    Args:
        sentence: a string representing a formula of propositional logic to be parsed for constructing a truth-table.

    Returns:
        a list of dictionaries of form {"input": (list of states), "label": (state) } representing positive examples to train (reinforce) on.
    """

    # TODO: parse sentence and return the number of atoms, and a 'ground_truth' function that returns the truth-value of the sentence given an assignment of truth-values to the atoms.
    length = None
    f = None

    examples = []
    combinations = product([True, False] for _ in range(length))
    for inputs in combinations:
        example = {
            "input": inputs,  # a list of states
            "label": f(inputs),  # a state
        }
        examples.append(example)
    return examples


def get_ssr_data(
    f: Callable[[bool, bool], bool] = lambda x, y: x and y
) -> list[dict[str, State]]:
    examples = []
    for p, q in product([True, False], [True, False]):
        example = {  # we join the states instead of passing a list
            "input": State("".join([str(int(p)), str(int(q))])),
            "label": State(str(int(f(p, q)))),
        }
        examples.append(example)
    return examples


def n_ary_data(n: int, connective: Callable[[bool, bool], bool] = lambda x, y: x and y):
    f = lambda inputs: reduce(connective, inputs)
    examples = []
    assignments = list(
        product([False, True], repeat=n)
    )  # get all possible combinations of truth values
    for inputs in assignments:
        example = {
            "input": [bool_to_state(atom) for atom in inputs],  # a list of states
            "label": bool_to_state(f(inputs)),  # a state
        }
        examples.append(example)
    return examples


def binary_data(
    f: Callable[[bool, bool], bool] = lambda x, y: x and y
) -> list[dict[str, State]]:
    """Given a boolean function, generate a dataset of positive examples to train agents on corresponding to the table representation of the function.

    Args:
        f: a boolean function

    Returns:
        a list of dictionaries of form {"state": ..., "act": ...} representing positive examples to train (reinforce) on.
    """
    examples = []
    for p, q in product([True, False], [True, False]):
        example = {
            "input": [State(str(int(p))), State(str(int(q)))],
            "label": State(str(int(f(p, q)))),
        }
        examples.append(example)
    return examples


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