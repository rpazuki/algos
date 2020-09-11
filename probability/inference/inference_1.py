from probability.empirical_distributions import FrequencyTable


class Binomial:
    def __init__(self, empirical_dist, level=None):
        if not isinstance(empirical_dist, FrequencyTable):
            raise ValueError(
                "'empirical_dist' argument must be a FrequencyTable class."
            )
        self.levels = empirical_dist.levels()
        if len(self.levels) != 2:
            raise ValueError(
                f"Binomial distribution has two levels, {len(self.levels)} is provided."
            )
        self.e_dist = empirical_dist
        #
        if level is None:
            self.theta_hat = self.e_dist.probability(self.levels[0])
        else:
            self.theta_hat = self.e_dist.probability(level)


class Multinomial:
    def __init__(self, empirical_dist):
        if not isinstance(empirical_dist, FrequencyTable):
            raise ValueError(
                "'empirical_dist' argument must be a FrequencyTable class."
            )
        self.e_dist = empirical_dist
        #

        self.theta_hat = {
            level: self.e_dist.probability(level) for i, level in enumerate(self.e_dist)
        }
