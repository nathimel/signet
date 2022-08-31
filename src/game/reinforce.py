"""Functions useful for training agents under simple reinforcement learning."""

import copy
import random
from typing import Any
import numpy as np


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


def construct_policy_maps(
    inputs: list, outputs: list
) -> dict[str, list[dict[str, Any]]]:
    """Construct the space of possible policy maps for Win-Stay-Lose-Shift with Inertia.

    Args:
        inputs: a list of hashable inputs

        ouputs: a list of hashable outputs

    Returns:
        a dict of the form {input1: [output1, output2, ...], ...}
    """
    # the carteisan product constructs the space of possible maps
    return {input_: [output for output in outputs] for input_ in inputs}


def update_map(
    maps: dict[Any, list[Any]],
    current_map: dict[Any, dict[str, Any]] = None,
    inertia: int = 1,
) -> dict[Any, dict[str, Any]]:
    """Shift from an unsuccesful map and update inertia counts appropriately."""
    if current_map is None:
        return {
            input_: {"output": shift_policy(maps[input_]), "losses": 0}
            for input_ in maps
        }

    new_map = {}
    for input_ in maps:
        output_dict = current_map[input_]
        if output_dict["losses"] > inertia:
            # shift
            new_output = shift_policy(maps[input_], output_dict["output"])
            new_map[input_] = {"output": new_output, "losses": 0}
        else:
            new_map[input_] = output_dict
    return new_map


def shift_policy(policies: list[Any], avoid: Any = None) -> dict[str, Any]:
    """Shift from an unsuccessful policy to any new one at random."""
    if avoid is not None:
        candidates = copy.deepcopy(policies)
        candidates.remove(avoid)
    else:
        candidates = policies
    return random.choice(candidates)
