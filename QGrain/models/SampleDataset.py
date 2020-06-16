__all__ = ["ClassesNotIncrementalError",
           "ClassesNotMatchError",
           "NaNError",
           "ArrayEmptyError",
           "SampleNameEmptyError",
           "DistributionSumError",
           "SampleDataset"]

from typing import Callable, Dict, Iterable, List, Tuple
from uuid import UUID

import numpy as np

from QGrain.models.SampleData import SampleData


class ClassesNotIncrementalError(Exception):
    """Raises while the array of grain size classes is not incremental.

    It's just an ASSUMPTION for the convenience of coding.
    """
    pass


class ClassesNotMatchError(Exception):
    """Raises while the grain size classes of a new batch of sample data are not equal to the existing classes."""
    pass


class NaNError(Exception):
    """Raises while there is at least one NaN value in the array."""
    pass


class ArrayEmptyError(Exception):
    """Raises while the length of array is 0."""
    pass


class SampleNameEmptyError(Exception):
    """Raises while the name of sample is empty."""
    pass


class DistributionSumError(Exception):
    """Raises while the sum of distribution array is not equal to 1 or 100."""
    pass


class SampleNotFountError(Exception):
    """Raises while the sample is not found in this dataset."""
    pass


class SampleDataset:
    def __init__(self):
        self.__classes = None
        self.__samples = {}

    @property
    def data_count(self) -> int:
        return len(self.__samples)

    @property
    def has_data(self) -> bool:
        if self.data_count == 0:
            return False
        else:
            return True

    @property
    def classes(self) -> np.ndarray:
        return self.__classes

    @property
    def samples(self) -> Iterable[SampleData]:
        return self.__samples.values()

    def is_incremental(self, nums: np.ndarray) -> bool:
        """Returns `True` while the array is incremental.
        This method is used to validate the array of grain size classes."""
        for i in range(1, len(nums)):
            if nums[i] <= nums[i-1]:
                return False
        return True

    def validate_classes(self, classes: np.ndarray):
        assert classes is not None
        assert type(classes) == np.ndarray
        assert classes.dtype == np.float64
        if len(classes) == 0:
            raise ArrayEmptyError()
        if np.any(np.isnan(classes)):
            raise NaNError(classes)
        # may raise when users select a wrong data file
        if not self.is_incremental(classes):
            raise ClassesNotIncrementalError(classes)

    def validate_sample_name(self, sample_name: str):
        assert sample_name is not None
        assert type(sample_name) == str
        if sample_name == "":
            raise SampleNameEmptyError()

    def validate_distribution(self, distribution: np.ndarray):
        assert distribution is not None
        assert type(distribution) == np.ndarray
        assert distribution.dtype == np.float64
        if len(distribution) == 0:
            raise ArrayEmptyError()
        if np.any(np.isnan(distribution)):
            raise NaNError(distribution)
        # check the sum is close to 1 or 100 (someone may use percentage)
        s = np.sum(distribution)
        if (s > 0.99 and s < 1.01) or (s > 99 and s < 101):
            np.true_divide(distribution, s, out=distribution)
        else:
            raise DistributionSumError(distribution)

    def add_batch(self, classes: np.ndarray, names: Iterable[str],
                  distributions: Iterable[np.ndarray]):
        self.validate_classes(classes)
        # If there is data already, it's necessary to check the consistency of classes.
        if self.has_data:
            equal_res = np.equal(self.classes, classes)
            if not np.all(equal_res):
                raise ClassesNotMatchError(self.classes, classes)

        assert len(names) == len(distributions)
        # use temp to implement roll-back
        temp = {}
        for name, distribution in zip(names, distributions):
            assert len(distribution) == len(classes)
            self.validate_sample_name(name)
            self.validate_distribution(distribution)
            sample = SampleData(name, classes, distribution)
            temp.update({sample.uuid: sample})
        # if no exception raised, the codes below will be executed
        self.__samples.update(temp)
        self.__classes = classes

    def combine(self, another_dataset):
        equal_res = np.equal(another_dataset.classes, self.classes)
        if np.all(equal_res):
            self.__samples.update(another_dataset.__samples)
        else:
            raise ClassesNotMatchError(self.classes, another_dataset.classes)

    def get_sample_by_id(self, uuid: UUID):
        if uuid in self.__samples:
            return self.__samples[uuid]
        else:
            raise SampleNotFountError("There is no sample with this id.", uuid)

    def remove_sample_by_id(self, uuid: UUID):
        try:
            return self.__samples.pop(uuid)
        except KeyError:
            raise SampleNotFountError("There is no sample with this id.", uuid)
