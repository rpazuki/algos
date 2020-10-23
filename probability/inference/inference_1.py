from abc import ABC, abstractmethod
from probability import Distribution
from probability.empirical_distributions import EmpiricalDistribution
from probability.empirical_distributions import FrequencyTable
from probability.empirical_distributions import DiscreteDistribution


class Inferrer(ABC):
    def __init__(self, empirical_distribution):
        if not isinstance(empirical_distribution, EmpiricalDistribution):
            raise ValueError(
                "The 'empirical_distribution' argument is not "
                " a  sub class of EmpiricalDistribution class."
            )
        self.__ed__ = empirical_distribution

    @classmethod
    def from_sample(cls, samples, name=None):
        ft = FrequencyTable(samples, name)
        return cls(ft)

    @classmethod
    def from_multilevels_sample(cls, samples, names=None):
        ft = DiscreteDistribution(samples, names)
        return cls(ft)

    @abstractmethod
    def probability(self, key):
        pass


class Multinomial(Distribution):
    def __init__(self, inferrer):

        self.inferrer = inferrer
        self.__ed__ = inferrer.__ed__

    def keys(self):
        return self.__ed__.keys()

    def items(self):
        return [(key, self.probability(key)) for key in self.keys()]

    def __getitem__(self, key):
        return self.probability(key)

    def get_random_variable(self):
        return self.__ed__.get_random_variable()

    def __contains__(self, key):
        return self.__ed__.__contains__(key)

    def probability(self, key):
        return self.inferrer.probability(key)

    def __rmul__(self, that):
        # Always rely on the left-multiplication
        return that.__mul__(self)

    def __mul__(self, that):
        pass
