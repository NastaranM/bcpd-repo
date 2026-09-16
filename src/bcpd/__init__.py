"""Recursive Bayesian changepoint detection.

Python port of the algorithm in Meftahi (2021), "Changepoint Detection in KPI
Time-Series Data - a Bayesian Approach", following Ruggieri & Lawrence (2012,
2014) and Ruggieri & Antonellis (2016).
"""

from .design import (
    harmonic_nested_sizes,
    joint_design,
    seasonal_design,
    trend_design,
)
from .detector import DetectionResult, detect, detect_multivariate
from .marginal import PriorSpec, SegmentLikelihood
from .multivariate import (
    MultivariateSegmentLikelihood,
    MVPriorSpec,
    log_multivariate_gamma,
)
from .recursion import ChangepointPosterior, forward_recursion, sample_changepoints

__version__ = "0.1.0"

__all__ = [
    "PriorSpec",
    "SegmentLikelihood",
    "forward_recursion",
    "sample_changepoints",
    "ChangepointPosterior",
    "trend_design",
    "seasonal_design",
    "joint_design",
    "harmonic_nested_sizes",
    "MVPriorSpec",
    "MultivariateSegmentLikelihood",
    "log_multivariate_gamma",
    "detect",
    "detect_multivariate",
    "DetectionResult",
]
