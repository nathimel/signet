"""File for creating the default agents, languages, and other data structures used in for predicting the truth values of boolean sentences with signaling networks."""

import numpy as np

from languages import State
from itertools import product
from typing import Any, Callable
from functools import reduce


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


def n_ary_data(
    n: int, 
    connective: Callable[[bool, bool], bool] = lambda x, y: x and y
    ):
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
