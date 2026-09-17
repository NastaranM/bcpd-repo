"""Exact Bayesian changepoint detection for time series.

Offline inference over the number and locations of changepoints by dynamic
programming, and online inference by particle filter, over a shared conjugate
segment model. Univariate or multivariate.

References
----------
Barry, D. & Hartigan, J. A. (1992). Product partition models for change point
problems. The Annals of Statistics 20(1), 260-279.
Fearnhead, P. (2006). Exact and efficient Bayesian inference for multiple
changepoint problems. Statistics and Computing 16(2), 203-213.
Fearnhead, P. & Liu, Z. (2007). On-line inference for multiple changepoint
problems. JRSS-B 69(4), 589-605.
Ruggieri, E. & Lawrence, C. E. (2014). The Bayesian change point and variable
selection algorithm. JCGS 23(1), 87-110.
"""

from .design import (
    constant_design,
    harmonic_nested_sizes,
    joint_design,
    seasonal_design,
    trend_design,
)
from .detector import DetectionResult, detect, detect_multivariate
from .marginal import PriorSpec, SegmentLikelihood
from .online import (
    OnlineChangepointFilter,
    OnlineResult,
    SegmentLengthPrior,
    optimal_resample,
)
from .multivariate import (
    MultivariateSegmentLikelihood,
    MVPriorSpec,
    log_multivariate_gamma,
)
from .recursion import (
    ChangepointPosterior,
    forward_recursion,
    log_n_configurations,
    sample_changepoints,
)

__version__ = "0.1.0"

__all__ = [
    "PriorSpec",
    "SegmentLikelihood",
    "forward_recursion",
    "sample_changepoints",
    "log_n_configurations",
    "ChangepointPosterior",
    "constant_design",
    "trend_design",
    "seasonal_design",
    "joint_design",
    "harmonic_nested_sizes",
    "MVPriorSpec",
    "MultivariateSegmentLikelihood",
    "log_multivariate_gamma",
    "detect",
    "detect_multivariate",
    "SegmentLengthPrior",
    "OnlineChangepointFilter",
    "OnlineResult",
    "optimal_resample",
    "DetectionResult",
]
