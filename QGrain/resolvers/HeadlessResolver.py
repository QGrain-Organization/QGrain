__all__ = ["FittingTask", "FinalLocalOptimizationError",
           "GlobalOptimizationError",
           "HeadlessResolver"]

from uuid import UUID, uuid4

import numpy as np
from scipy.optimize import OptimizeResult

from QGrain.algorithms import DistributionType
from QGrain.models.AlgorithmSettings import AlgorithmSettings
from QGrain.models.FittingResult import FittingResult
from QGrain.models.SampleData import SampleData
from QGrain.resolvers.Resolver import Resolver


class FittingTask:
    def __init__(self, sample: SampleData,
                 distribution_type=DistributionType.GeneralWeibull,
                 component_number=3,
                 algorithm_settings=None):
        self.uuid = uuid4()
        self.sample = sample
        self.component_number = component_number
        self.distribution_type = distribution_type
        if algorithm_settings is None:
            self.algorithm_settings = AlgorithmSettings()
        else:
            self.algorithm_settings = algorithm_settings


class FinalLocalOptimizationError(Exception):
    """
    Raises when the final local optimization failed.
    """
    pass


class GlobalOptimizationError(Exception):
    """
    Raises when the global optimization failed.
    """
    pass


class HeadlessResolver(Resolver):
    def __init__(self):
        super().__init__()
        self.current_task = None # type: FittingTask
        self.current_result = None # type: FittingResult
        self.current_exception = None # type: Exception

    def on_fitting_succeeded(self, algorithm_result: OptimizeResult):
        result = self.get_fitting_result(algorithm_result.x)
        self.current_result = result

    def on_global_fitting_failed(self, algorithm_result: OptimizeResult):
        self.current_exception = GlobalOptimizationError(algorithm_result.message)

    def on_final_fitting_failed(self, algorithm_result: OptimizeResult):
        self.current_exception = FinalLocalOptimizationError(algorithm_result.message)

    def on_exception_raised_while_fitting(self, exception: Exception):
        self.current_exception = exception

    def execute_task(self, task: FittingTask):
        self.current_task = task
        self.distribution_type = task.distribution_type
        self.component_number = task.component_number
        if task.algorithm_settings is not None:
            self.change_settings(task.algorithm_settings)
        self.feed_data(task.sample)
        self.try_fit()
        if self.current_result is None:
            assert self.current_exception is not None
            return False, self.current_task, self.current_exception
        else:
            return True, self.current_task, self.current_result

