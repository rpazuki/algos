from abc import abstractmethod
from collections import Counter
import numpy as np
from probability2 import Key
from probability2 import Distribution


class EmpiricalDistribution(Distribution):
    def __init__(self, samples):
        """Construct an abstract distribution and count the number of
        occurenc of items in the samples.

        Args:
            samples (iterable):
                An iterable that contains the observed samples
                or a dictionary of (key:frequency).

        Raises:
            ValueError: Raises when the provided sample is None.
        """
        super().__init__()
        if samples is None:
            raise ValueError("samples argument is None.")

        self._counter = Counter(samples)
        # Elements count
        self.total = sum(self._counter.values())

    def _check_keys_consistencies_(self):
        rv_len = self.get_random_variable().size

        def compare_single_elements():
            try:
                keys = iter(self.keys())
                first_row = next(keys)
            except StopIteration:  # Empty rows
                return
            first_row_type = type(first_row)
            if isinstance(first_row, tuple):
                # For single elements that are tuple
                # we check both type and length
                first_row_len = len(first_row)
                for row in keys:
                    if not isinstance(row, first_row_type):
                        raise ValueError(
                            "The type of the 'factors' are not consistence."
                        )
                    if len(row) != first_row_len:
                        raise ValueError(
                            "The length of the 'factors' are not consistence."
                        )
            else:  # For other single elements, we just
                # check the type
                for row in keys:
                    if not isinstance(row, first_row_type):
                        raise ValueError(
                            "The type of the 'factors' are not consistence."
                        )

        def compare_multilevel_elements():
            # We suppose the keys were tuples
            # and each Random Variable (RV) is positioned
            # in a fix place of the n-tuple.
            # Therefore, the levels of the RV can be
            # found by iterating over each tuple's item
            # Convert each features line to tuple
            tuples = (Key(row) for row in self.keys())
            try:
                first_row = next(tuples)
            except StopIteration:  # Empty rows
                return
            first_row_len = len(first_row)
            first_row_types = [type(item) for item in first_row]
            for row in tuples:
                # compair length
                if len(row) != first_row_len:
                    raise ValueError("The length of the 'factors' are not consistence.")
                # compair row's elements type
                comparisions = [
                    isinstance(element, type_1)
                    for element, type_1 in zip(row, first_row_types)
                ]
                if not all(comparisions):
                    raise ValueError("The types of the 'factors' are not consistence.")

        if rv_len > 1:
            compare_multilevel_elements()
        else:
            compare_single_elements()

    @staticmethod
    def digitize(samples, start, stop, num=10, endpoint=True, right=False, levels=None):
        """[summary]

        Args:
            samples (numeric array):
                continouse values that needs digitization.
            start (numeric):
                The starting value of the sequence.
            stop (numeric):
                The end value of the sequence, unless `endpoint` is set to False.
                In that case, the sequence consists of all but the last of ``num + 1``
                evenly spaced samples, so that `stop` is excluded.  Note that the step
                size changes when `endpoint` is False.
            num (int, optional):
                Number of samples to generate. Default is 10. Must be non-negative.
            endpoint (bool, optional):
                If True, `stop` is the last sample. Otherwise, it is not included.
                Defaults to True.
            right (bool):
                Indicating whether the intervals include the right or the left bin
                edge. Default behavior is (right==False) indicating that the interval
                does not include the right edge. The left bin end is open in this
                case, i.e., bins[i-1] <= x < bins[i] is the default behavior for
                monotonically increasing bins.
            levels (list, optional):
                List of labels for each step. Defaults to None.

        Returns:
            numpy array:
                [type]: An array of bins/levels of digitized samples.
        """
        bins = np.linspace(start, stop, num, endpoint=endpoint)
        return EmpiricalDistribution.digitize_bin(samples, bins, right, levels)

    @staticmethod
    def digitize_bin(samples, bins, right=False, levels=None):
        """Return the digitized samples by returning the coresponding bins value.
           When 'levels' is provided, bins values replace by levels.

           =========  =============  ============================
           `right`    order of bins  returned index `i` satisfies
           =========  =============  ============================
           ``False``    increasing   ``bins[i-1] <= x < bins[i]``
           ``True``     increasing   ``bins[i-1] < x <= bins[i]``
           ``False``    decreasing   ``bins[i-1] > x >= bins[i]``
           ``True``     decreasing   ``bins[i-1] >= x > bins[i]``
           =========  =============  ============================

        Args:
            samples (numeric array):
                continouse values that needs digitization.
            bins (array):
                Array of bins. It has to be 1-dimensional and monotonic.
            right (bool):
                Indicating whether the intervals include the right or
                the left bin edge. Default behavior is (right==False)
                indicating that the interval does not include the right
                edge. The left bin end is open in this case, i.e.,
                bins[i-1] <= x < bins[i] is the default behavior for
                monotonically increasing bins.
            levels (list, optional):
                List of labels for each step. Defaults to None.

        Raises:
            ValueError:
                Raises when the length of levels is not equal to
                the length of bins minus one.

        Returns:
            numpy array:
                An array of bins/levels of digitized samples.
        """
        if levels is not None and len(levels) != len(bins) - 1:
            raise ValueError(
                f"'levels' length ({len(levels)}) is not "
                f"equal to bins length-1 ({len(bins)-1})."
            )
        indices = np.digitize(samples, bins, right)

        if levels is None:
            # Extend the bins to include left outliers
            delta_left = bins[1] - bins[0]
            bins_extended = np.r_[[bins[0] - delta_left], bins]
            return bins_extended[indices]

        # Extend the levels to include outliers
        levels_extended = np.r_[["less"], levels, ["more"]]
        return levels_extended[indices]

    def normalise(self):
        """Normalise the distribution."""
        for k in self._counter:
            self._counter[k] = self._counter[k] / self.total
        self.total = 1.0

    def probability(self, key):
        """Gets the probability of the random variable, when its value is 'key'.

           It return zero if the value is not observed.

        Args:
            key (object):
                the value of the random variable.

        Returns:
            float: probability of the random variable.
        """
        if self.total == 0:
            return 0

        return self.__getitem__(key) / self.total

    def freq(self, *args, **kwargs):
        key = self.get_random_variable().to_key(*args, **kwargs)
        return self.frequency(key, normalised=False)

    def frequency(self, key, normalised=False):
        """Gets the frequency of the random variable, when its value is 'key'.

           It return zero if the value is not observed.

        Args:
            key (object):
                the value of the random variable.
            normalised (bool, optional):
                normalize the return. Defaults to False.

        Returns:
            int or float: frequency or probability of the random variable.
        """
        if self.total == 0:
            return 0

        if normalised:
            return self._counter[key] / self.total
        else:
            return self._counter[key]

    def keys(self):
        return self._counter.keys()

    def items(self):
        return self._counter.items()

    def frequencies(self, normalised=True):
        """A list of frequencies of class occurenc.

        Args:
            normalised (bool, optional):
                The normalisation flag. Defaults to True.

        Returns:
            list: A list of floats or integers, depending of normalisation.
        """
        if self.total == 0:
            return np.zeros(len(self._counter.keys()))

        values = np.array(list(self._counter.values()))

        if normalised:
            return np.array(values) / self.total

        return np.array(values)

    def keys_as_list(self):
        return [k for k in self._counter.keys()]

    def most_common(self, num: int = None):
        """List the n most common elements and their counts from the most
            common to the least. If n is None, then list all element counts.

        Args:
            num (int, optional):
                The maximum length of the returned list. Defaults to None.

        Returns:
            list: A list of tuples. The first element of the tuple is a class
                  key and th second one is its count.
        """
        return self._counter.most_common(num)

    @abstractmethod
    def get_random_variable(self):
        pass

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def to_table(self, normalised=False, sort=False):
        pass

    @abstractmethod
    def __mul__(self, that):
        pass

    @abstractmethod
    def __rmul__(self, that):
        pass

    def __getitem__(self, key):
        """An indexer that returns the count of the class key.

            Returns zero if the 'key' does not exist in samples or
            the samples iterator is empty.


        Args:
            key (object):
                The key that specifies the class name in samples.

        Returns:
            [float]: The count of occurrence of class 'key'.
                     Returns zero if the 'key' does not exist in samples or
                     the samples iterator is empty.
        """
        if self.total == 0:
            return 0

        if isinstance(key, slice):
            return list(self._counter.items())[key]

        return self._counter[key]

    def __contains__(self, key):
        return key in self._counter
