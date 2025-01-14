from typing import Callable


def newton_raphson(f: Callable, f_prime: Callable, x0: float, tol: float, 
                   max_iter: int, return_list: bool = False) -> float:
    """
    This is a well-known root-finding algorithm of the 
    equation f(x) = 0, for some function f

    Arguments:
    ----------
    f : Callable
        Some function.S
    f_prime : Callable
        The first derivative of f.
    tol : float
        Some tolerance level epsilon to reach.
    max_iter : int
        The maximum number of iterations.
    x0 : float
        Initial guess of the root.

    Returns: 
    --------
    float
        An approximation of the root.
    """

    if return_list:
        x0_list = [x0]
        for _ in range(max_iter):
            x_new = x0 - f(x0) / f_prime(x0)
            x0_list.append(x_new)
            if abs(x0 - x_new) < tol:
                break
            x0 = x_new
        return x0_list
    else:
        for _ in range(max_iter):
            x_new = x0 - f(x0) / f_prime(x0)
            if abs(x0 - x_new) < tol:
                break
            x0 = x_new
        return x0


