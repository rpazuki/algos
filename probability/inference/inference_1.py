from probability.empirical_distributions import FrequencyTable


class Binomial(FrequencyTable):
    def __init__(self, samples, level=None, name="X1", check_keys_consistencies=True):
        super().__init__(samples, name, check_keys_consistencies)
        levels = self.levels()
        if len(levels) != 2:
            raise ValueError(
                f"Binomial distribution has two levels, {len(levels)} is provided."
            )
        #
        if level is None:
            self.theta_hat = self.probability(levels[0])
        else:
            self.theta_hat = self.probability(level)


class Multinomial(FrequencyTable):
    def __init__(self, samples, name="X1", check_keys_consistencies=True):
        super().__init__(samples, name, check_keys_consistencies)
        #
        self.theta_hat = {
            level: self.probability(level) for i, level in enumerate(self.levels())
        }
