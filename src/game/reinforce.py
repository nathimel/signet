"""Functions useful for training agents under simple reinforcement learning."""

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


##############################################################################
# Reinforcement learning algorithms
##############################################################################


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
) -> np.ndarray:
    """Win-stay lose-shift learning with inertia.

    Requires a history to determine the number of losses, and to switch if greater than $i$, the inertia.
    """
