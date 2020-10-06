from probability import Distribution
from probability import DiscreteRV
from probability.empirical_distributions import FrequencyTable


class AbstractNominal(Distribution):
    def __init__(self, levels, name="X1"):
        self.name = name
        self.levels = list(levels)
        self.discrete_rv = DiscreteRV(name, levels[0])

    def keys(self):
        return self.levels

    def items(self):
        return [(key, self.probability(key)) for key in self.levels]

    def _get_random_variable_(self):
        return self.discrete_rv

    def __getitem__(self, key):
        return self.probability(key)

    def __contains__(self, key):
        return key in self.levels


class Binomial(AbstractNominal):
    def __init__(self, theta, levels, name="X1"):
        super().__init__(levels, name)
        if len(levels) != 2:
            raise ValueError(
                f"Binomial distribution has two levels, {len(levels)} is provided."
            )
        if theta < 0 or theta > 1:
            raise ValueError(f"'theta' must be in [0, 1] ({theta}).")

        self.theta = theta

    @classmethod
    def from_frequency_table(cls, frequency_table, level=None, name=None):
        if not isinstance(frequency_table, FrequencyTable):
            raise ValueError(
                "The 'frequency_table' argument is not " " a FrequencyTable class."
            )
        levels = frequency_table.levels()
        if len(levels) != 2:
            raise ValueError(
                f"Binomial distribution has two levels, {len(levels)} is provided."
            )
        #
        if level is None:
            level = levels[0]

        if name is None:
            name = frequency_table.name
        # Maximum Likelihood estimation
        theta_hat = frequency_table.probability(level)
        return cls(theta_hat, levels, name)

    @classmethod
    def from_sample(cls, samples, level=None, name=None):
        ft = FrequencyTable(samples, name)
        return Binomial.from_frequency_table(ft, level, name)

    def normalise(self):
        return

    def probability(self, key):
        if key == self.levels[0]:
            return self.theta
        if key == self.levels[1]:
            return 1 - self.theta

        return 0

    def __mul__(self, that):
        pass

    def __rmul__(self, that):
        pass


class Multinomial(AbstractNominal):
    def __init__(self, thetas, levels, name="X1"):
        super().__init__(levels, name)
        for theta in thetas:
            if theta < 0 or theta > 1:
                raise ValueError(f"'theta' must be in [0, 1] ({theta}).")

        self.thetas = thetas

    @classmethod
    def from_frequency_table(cls, frequency_table, name=None):
        if not isinstance(frequency_table, FrequencyTable):
            raise ValueError(
                "The 'frequency_table' argument is not " " a FrequencyTable class."
            )
        levels = frequency_table.levels()

        if name is None:
            name = frequency_table.name
        # Maximum Likelihood estimation
        theta_hats = [frequency_table.probability(level) for level in levels]
        return cls(theta_hats, levels, name)

    @classmethod
    def from_sample(cls, samples, name=None):
        ft = FrequencyTable(samples, name)
        return Multinomial.from_frequency_table(ft, name)

    def normalise(self):
        theta_sum = sum(self.thetas)
        self.thetas = [theta / theta_sum for theta in self.thetas]

    def probability(self, key):
        indices = [i for i, k in enumerate(self.levels) if k == key]
        if len(indices) == 0:
            return 0

        return self.thetas[indices[0]]

    def __mul__(self, that):
        pass

    def __rmul__(self, that):
        pass
