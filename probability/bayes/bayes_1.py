from probability.empirical_distributions import FrequencyTable


class Binomial(FrequencyTable):
    def __init__(self, samples, alpha=1, beta=1, level=None, name="X1", consis=True):
        super().__init__(samples, name, consis)

        levels = self.levels()
        if len(levels) != 2:
            raise ValueError(
                f"Binomial distribution has two levels, {len(levels)} is provided."
            )
        #
        self.alpha = alpha
        self.beta = beta
        if level is None:
            self.n_level = self.frequency(levels[0], normalised=False)
        else:
            self.n_level = self.frequency(level, normalised=False)

    def map(self):

        # (n + \alpha)/ (\alpha + \beta + N)
        return (self.n_level + self.alpha) / (self.alpha + self.beta + self.total)


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
