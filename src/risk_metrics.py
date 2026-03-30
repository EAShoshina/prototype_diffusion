"""
Tail risk metrics: Value at Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall).
"""
import numpy as np


def compute_var(returns: np.ndarray, alpha: float = 0.95) -> float:
    """
    Compute Value at Risk (VaR) at confidence level alpha.

    VaR_alpha = inf{ l : P(X <= l) >= alpha }

    Args:
        returns: 1D array of simulated return scenarios
        alpha: Confidence level, e.g. 0.95

    Returns:
        VaR value (negative = loss)
    """
    return float(np.quantile(returns, 1 - alpha))


def compute_cvar(returns: np.ndarray, alpha: float = 0.95) -> float:
    """
    Compute Conditional VaR (CVaR / Expected Shortfall) at level alpha.

    CVaR_alpha = E[ X | X <= VaR_alpha ]

    Args:
        returns: 1D array of simulated return scenarios
        alpha: Confidence level

    Returns:
        CVaR value (mean of worst (1-alpha) scenarios)
    """
    var = compute_var(returns, alpha)
    tail = returns[returns <= var]
    return float(np.mean(tail)) if len(tail) > 0 else var


def risk_report(returns: np.ndarray, alphas=(0.95, 0.99)) -> dict:
    """
    Compute VaR and CVaR for multiple confidence levels.

    Args:
        returns: 1D array of return scenarios
        alphas: Tuple of confidence levels

    Returns:
        Dict with keys like 'VaR_0.95', 'CVaR_0.95', etc.
    """
    report = {}
    for a in alphas:
        report[f"VaR_{a}"] = compute_var(returns, a)
        report[f"CVaR_{a}"] = compute_cvar(returns, a)
    return report
