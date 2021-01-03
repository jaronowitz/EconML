# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
from sklearn.linear_model import LinearRegression
from econml.dml import LinearDML
import numpy as np


class TestMonteCarlo(unittest.TestCase):

    def test_montecarlo(self):
        """Test that we can perform nuisance averaging, and that it reduces the variance in a simple example."""
        y = np.random.normal(size=30) + [0, 1] * 15
        T = np.random.normal(size=(30,)) + y
        W = np.random.normal(size=(30, 3))
        est1 = LinearDML(model_y=LinearRegression(), model_t=LinearRegression())
        est2 = LinearDML(model_y=LinearRegression(), model_t=LinearRegression(), monte_carlo_iterations=2)
        # Run ten experiments, recomputing the variance of 10 estimates of the effect in each experiment
        v1s = [np.var([est1.fit(y, T, W=W).effect() for _ in range(10)]) for _ in range(10)]
        v2s = [np.var([est2.fit(y, T, W=W).effect() for _ in range(10)]) for _ in range(10)]
        # The average variance should be lower when using monte carlo iterations
        assert np.mean(v2s) < np.mean(v1s)
