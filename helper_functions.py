import numpy as np
from typing import Any

def magnitude(n: float) -> int:
    """Returns the order of magnitude of n.

    Parameters
    ----------
    1. n : float
        - A number.
    Returns
    -------
    - int
        - Order of magnitude of n.
    """
    return int(np.ceil(np.log10(np.abs(n))))


def format_sig_figs(n: float | list[float], sig_figs: int, exponential: bool = False) -> str | list[str]:
    """Returns a number or list of numbers formatted as strings with the desired
    number of significant figures.

    Parameters
    ----------
    1. n : Union[float,List]
        - A single number or an iterable of numbers to be formatted.
    2. sig_figs : int
        - The desired number of significant figures.
    3. exponential : bool (default False)
        - If True, numbers with greater than `sig_figs` digits will be formatted
          as exponential.
    Returns
    -------
    - Union[str,List[str]]
        - The formatted number string, or a list of formatted number strings if
          a list was input.

    Raises
    ------
    - ValueError
        - If a non-integer is given as `sig_figs`.
    """
    if type(sig_figs) is not int:
        raise ValueError("sig_figs must be an integer.")
    if isinstance(n, float):
        mag = magnitude(n)
        if exponential and mag > sig_figs:
            fmt = f"{{:.{sig_figs}e}}"
        else:
            decimals = max(0, sig_figs - mag)
            fmt = f"{{:.{decimals}f}}"
        return fmt.format(n)
    else:
        if False not in [isinstance(x, float) for x in n]:
            return [format_sig_figs(x, sig_figs, exponential=exponential) for x in n if type(x) is float]  # type: ignore[misc]
        else:
            raise ValueError("`n` must only contain floats.")


def is_numeric(n: Any) -> bool:
    return isinstance(n, float) or isinstance(n, int)


