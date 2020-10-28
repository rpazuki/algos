from collections import Counter
import numpy as np
from probability.core import RowKey
from probability.core import Table


class FrequencyTable(Table):
    def __init__(self, samples, names=None, consistencies=True, _internal_=False):
        counter = Counter(samples)
        super().__init__(counter, names, _internal_)
        # Elements count
        try:
            self.total = sum(counter.values())
        except TypeError:
            self.total = 1
        #
        if consistencies:
            self._check_keys_consistencies_()

    @classmethod
    def from_np_array(cls, samples, names=None):
        """Construct a FrequencyTable from a 2d numpy array or list of lists.
           The resulting keys are tuples.

        Args:
            samples (list or numpy.ndarray):
                the observed samples.
            names (list, optional):
                List of names of the columns.
                If it is not provided, it creates as 'Xn'.
                Defaults to None.

        Raises:
            ValueError: Raises when the samples argument is not list or
                        numpy.ndarray.
        """
        if not isinstance(samples, (np.ndarray, list)):
            raise ValueError(
                "'sample' argument must be numpy 2D ndarray or list of list."
            )
        elif isinstance(samples, list) and not isinstance(samples[0], list):
            raise ValueError(
                "'sample' argument must be numpy 2D ndarray or list of list."
            )
        # Convert rows to element, before calling
        # the construct
        return cls(samples=[RowKey(row) for row in samples], names=names)

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
        return FrequencyTable.digitize_bin(samples, bins, right, levels)

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

    def __getitem__(self, args):
        value = super().__getitem__(args)
        if value is None:
            return 0

        return value

    def freq(self, *args, **kwargs):
        key = self.columns.to_key(*args, **kwargs)
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

        value = self.__getitem__(key)

        if normalised:
            return value / self.total
        else:
            return value

    def prob(self, *args, **kwargs):
        key = self.columns.to_key(*args, **kwargs)
        return self.probability(key)

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

    def normalise(self):
        """Normalise the distribution."""
        for k in self.keys():
            self[k] = self[k] / self.total
        self.total = 1.0

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
        return Counter(self).most_common(num)

    def summary(self):
        return (
            "Frequency Table \n"
            f"Column names:'{self.names}'\n"
            f"total:{self.total}\n"
            f"normalised:{np.abs(self.total -1) <= 1e-16}\n"
        )

    def marginal(self, *args):
        """Marginalize the Table over a set of columns.

        Args:
            args (list):
                List of column names to marginalised.

        Raises:
            ValueError:
                Raises when one of the column names is
                not defined.
                Or raises when requested for all column names.

        Returns:
            Table: A new marginalised Table.
        """
        (rows, names) = self._group_by_(*args)
        return FrequencyTable(rows, names, consistencies=False, _internal_=True)

    def condition_on(self, *args):
        (rows, names, children_names) = self._group_on_(*args)
        return FrequencyTable(
            {
                key: FrequencyTable(
                    values, children_names, consistencies=False, _internal_=True
                )
                for key, values in rows.items()
            },
            names,
            consistencies=False,
            _internal_=True,
        )

    def __mul__(self, right):
        (rows, names) = self._product_(right)
        return FrequencyTable(rows, names, consistencies=False, _internal_=True)

    def __rmul__(self, left):
        (rows, names) = left._product_(self)
        return FrequencyTable(rows, names, consistencies=False, _internal_=True)
