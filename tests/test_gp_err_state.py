import numpy as np
import GP  # noqa: F401 - triggers __init__ side effects


def test_numpy_err_state_overflow_ignored():
    err = np.geterr()
    assert err["over"] == "ignore"
    assert err["invalid"] == "ignore"
    assert err["divide"] == "ignore"

