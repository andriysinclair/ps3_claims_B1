import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)
    winsorizer = Winsorizer(lower_quantile=lower_quantile, upper_quantile=upper_quantile)
    transformed = winsorizer.fit_transform(X)

    lower_bound = np.quantile(X, lower_quantile)
    upper_bound = np.quantile(X, upper_quantile)

    assert X.size > 0, "Input array X must not be empty"
    assert len(transformed) == len(X), "Output length does not match input length"

    assert np.all(transformed >= lower_bound), f"Values below {lower_bound} found"
    assert np.all(transformed <= upper_bound), f"Values above {upper_bound} found"
