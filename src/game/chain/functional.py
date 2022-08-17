from languages import State

Example = dict[str, list[State]]

def generate_data(length: int, f = lambda x: x) -> list[Example]:
    """Generate synthetic data defining a distribution p(x,y) corresponding to an operation on bitstrings.

    Args:
        length: the length of bitstrings to generate data for.

        f: the function on bitstrings to generate data for. Default is identity.
    """
