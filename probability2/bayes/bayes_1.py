from probability2 import Distribution
from probability2 import DiscreteRV
from probability2.empirical_distributions import FrequencyTable


class Binomial(Distribution):
    def __init__(self, samples, alpha=1, beta=1, level=None, name="X1", consis=True):
        super().__init__(samples)
        #
        # Random varable's details
        try:
            first_row = next(iter(self.keys()))
        except StopIteration:
            first_row = None

        self.discrete_rv = DiscreteRV(name, first_row)
        self.name = name
        #
        levels = self.levels()
        if len(levels) != 2:
            raise ValueError(
                f"Binomial distribution has two levels, {len(levels)} is provided."
            )
        #
        self.alpha = alpha
        self.beta = beta
        if level is None:
            self.level = levels[0]
        else:
            self.level = level

        self.n_level = self.frequency(self.level, normalised=False)
        # (n + \alpha)/ (\alpha + \beta + N)
        self.map = (self.n_level + self.alpha) / (self.alpha + self.beta + self.total)
        self.normalise()
        #
        if consis:
            self._check_keys_consistencies_()

    def probability(self, key):
        """Gets the probability of the random variable, when its value is 'key'.

           It return zero if the value is not observed.

        Args:
            key (object):
                the value of the random variable.

        Returns:
            float: probability of the random variable.
        """
        if key == self.level:
            return self.map
        elif key in self:
            return 1 - self.map

        return 0.0

    def summary(self):
        pass

    def to_table(self, normalised=False, sort=False):
        pass

    def __mul__(self, that):
        pass

    def __rmul__(self, that):
        pass

    def _get_random_variable_(self):
        return self.discrete_rv

    def __getitem__(self, key):
        """An indexer that returns the count of the class key.

            Returns zero if the 'key' does not exist in samples or
            the samples iterator is empty.


        Args:
            key (object):
                The key that specifies the class name in samples.

        Returns:
            [float]: The probability of class 'key'.
                     Returns zero if the 'key' does not exist in samples or
                     the samples iterator is empty.
        """

        if isinstance(key, slice):
            keys = list(self.keys())[key]
            return [self.probability(k) for k in keys]

        return self.probability(key)

    def __str__(self):
        return (
            f"Binomial distribution (rv:'{self.discrete_rv.name}', "
            f" alpha:{self.alpha}, beta:{self.beta})"
        )

    __repr__ = __str__


class Multinomial(FrequencyTable):
    def __init__(self, samples, alphas, name="X1", check_keys_consistencies=True):
        super().__init__(samples, name, check_keys_consistencies)
        #
        self.alphas = alphas

    def map(self):
        sum_alpha = sum([(alpha - 1) for alpha in self.alphas])

        def theta_hat(i, level):
            # theta_hat_i = (n_i + \alpha_i - 1)/(N + \sum_j (\alpha_j - 1))
            n_i = self.frequency(level, normalised=False)
            alpha_i = self.alphas[i]
            return (n_i + alpha_i - 1) / (self.total + sum_alpha)

        return {level: theta_hat(i, level) for i, level in enumerate(self.levels())}
