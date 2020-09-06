import numpy as np
from probability.distributions import Key
from probability.distributions import MultiDiscreteRV


class ConditionalDistribution:
    def __init__(self, distributions, conditional_names):
        """Create a conditional distributions.

        Args:
            distributions (dict): a dictionary of (key:DiscreteDistribution)
                                  where key is the value of the random variables
                                  that is conditioned on.
            conditional_names ([type]): [description]
        """
        self.conditional_rvs = MultiDiscreteRV(
            list(distributions.keys()), conditional_names
        )
        # For each conditional levels, we store its equivalent dist.
        self.distributions = distributions
        #
        first_key = next(iter(distributions))
        first_example_dist = self.distributions[first_key]
        self.rvs = first_example_dist.rvs
        self.names = first_example_dist.names

    def probability(self, key, conditional_key):
        if conditional_key not in self.distributions:
            return 0
        else:
            return self.distributions[conditional_key].probability(key)

    def frequency(self, key, conditional_key, normalised=False):
        if conditional_key not in self.distributions:
            return 0
        else:
            return self.distributions[conditional_key].frequency(key, normalised)

    def summary(self):
        return (
            "Discrete conditional distribution \n"
            f"conditioned on: {self.conditional_rvs.names}\n"
            f"conditioned levels: {self.conditional_rvs.levels}\n"
            f"random variables:'{self.names}'\n"
            f"levels: {self.rvs}"
        )

    def marginal(self, by_names):
        """Marginalize the distribution over a set of random variables.

        Args:
            by_names (list): List of variable names to marginalised.

        Raises:
            ValueError:  Raises when one of the random variable names is
                        not defined in rvs.

        Returns:
            DiscreteConditionalDistribution: A new marginalised distribution.
        """
        for name in by_names:
            if name not in self.rvs:
                raise ValueError(
                    f"Random variable {name} is not defined."
                    "(Maybe it is a conditional one?)"
                )
        new_distributions = {}
        for conditional_key, distribution in self.items():
            new_distributions[conditional_key] = distribution.marginal(by_names)

        return ConditionalDistribution(new_distributions, self.conditional_rvs.names)

    def condition_on(self, on_names):
        for name in on_names:
            if name not in self.rvs:
                raise ValueError(
                    f"Random variable {name} is not defined."
                    "(Maybe it is a conditional one?)"
                )
        # per each conditional key, we must create a
        # new conditioned distribution
        new_distributions = {}
        for conditional_key, distribution in self.items():
            # here, the new conditioned distribution per key is made
            conditioned_one = distribution.condition_on(on_names)
            # loop over all newely conditioned distribution's distributions
            # and combine their conditional keys with the self distribution
            # to create a new distributions dictionary
            for new_key, new_distribution in conditioned_one.items():
                combined_key = Key(conditional_key) * Key(new_key)
                new_distributions[combined_key] = new_distribution

        return ConditionalDistribution(
            new_distributions, np.r_[self.conditional_rvs.names, on_names]
        )

    def keys(self):
        return self.distributions.keys()

    def items(self):
        return self.distributions.items()

    def __getitem__(self, key):
        return self.distributions[key]

    def __contains__(self, key):
        return key in self.distributions

    def __iter__(self):
        return iter(self.distributions.keys())

    def __str__(self):
        return (
            "Discrete Conditiona Distribution ("
            f"conditioned on: {self.conditional_rvs.names}"
            f", rvs: {self.rvs.names})"
        )

    def __repr__(self):
        return self.__str__()
