"""File for creating the default agents, languages, and other data structures used in for predicting the truth values of boolean sentences with signaling networks."""

import numpy as np

from network import SignalTree
from languages import Signal, SignalMeaning, SignalingLanguage, State, StateSpace
from agents import (
    AttentionAgent,
    Compressor,
    Receiver,
    ReceiverModule,
    ReceiverSender,
    Sender,
    SenderModule,
)
from itertools import product
from typing import Any, Callable
from functools import reduce

##############################################################################
# Data and evaluation
##############################################################################


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


def n_ary_data(n: int):
    f = lambda inputs: reduce(lambda x, y: x and y, inputs)
    examples = []
    assignments = list(
        product([0, 1], repeat=n)
    )  # get all possible combinations of truth values
    for inputs in assignments:
        example = {
            "input": [State(str(atom)) for atom in inputs],  # a list of states
            "label": State(str(f(inputs))),  # a state
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


def empirical_accuracy(
    net: SignalTree, dataset: list[dict[str, Any]], num_rounds: int = None
) -> float:
    """
    Evaluate the accuracy of a signaling networks by computing the average accuracy on the dataset.

    Args:
        num_rounds: an int representing how many interactions to record.
    """
    if num_rounds is None:
        num_rounds = len(dataset)

    num_correct = 0
    for _ in range(num_rounds):
        example = np.random.choice(dataset)

        x = example["input"]
        y = example["label"]
        y_hat = net(x)

        num_correct += 1 if y_hat == y else 0
        net.update()

    return num_correct / num_rounds


##############################################################################
# Utility functions
#
# for creating the default agents, languages, and other data structures used
# in predicting the truth values of boolean sentences with signaling networks
##############################################################################


def get_binary_language() -> SignalingLanguage:
    """Get a 2 state, 2 signal SignalingLanguage instance initialized for boolean games."""

    states = [State(name="0"), State(name="1")]
    universe = StateSpace(states)

    dummy_meaning = SignalMeaning(
        states=states,
        universe=universe,
    )
    signals = [
        Signal(form="0", meaning=dummy_meaning),
        Signal(form="1", meaning=dummy_meaning),
    ]

    return SignalingLanguage(
        signals=signals,
    )


def get_quaternary_language() -> SignalingLanguage:
    """Get a 2 state, 4 signal SignalingLanguage instance initialized for boolean games."""
    states = [State(name="0"), State(name="1")]
    universe = StateSpace(states)
    dummy_meaning = SignalMeaning(
        states=states,
        universe=universe,
    )
    signals = [
        Signal(form="00", meaning=dummy_meaning),
        Signal(form="01", meaning=dummy_meaning),
        Signal(form="10", meaning=dummy_meaning),
        Signal(form="11", meaning=dummy_meaning),
    ]

    return SignalingLanguage(
        signals=signals,
    )


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


# def get_compressor(input_size: int) -> Compressor:
#     """Get a Compressor instance initialized for boolean games."""
#     return Compressor(
#         attention_1=AttentionAgent(input_size),
#         attention_2=AttentionAgent(input_size),
#         receiver_sender=get_receiver_sender(),
#     )

# def get_input_agent(input_size: int) -> InputSender:
#     """Get an input agent (Sender) instance initialized for boolean games."""
#     return InputSender(input_size)

# def get_output_agent() -> OutputReceiver:
#     """Get an output agent (Receiver) instance initialized for boolean games."""
#     return OutputReceiver(2)
