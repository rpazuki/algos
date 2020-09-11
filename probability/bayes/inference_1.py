from probability.empirical_distributions import FrequencyTable


class Binomial:
    def __init__(self, empirical_dist, alpha=1, beta=1, level=None):
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
        self.alpha = alpha
        self.beta = beta
        if level is None:
            self.n_level = self.e_dist.frequency(self.levels[0], normalised=False)
        else:
            self.n_level = self.e_dist.frequency(level, normalised=False)

    def map(self):

        # (n + \alpha)/ (\alpha + \beta + N)
        return (self.n_level + self.alpha) / (
            self.alpha + self.beta + self.e_dist.total
        )


class Multinomial:
    def __init__(self, empirical_dist, *args):
        if not isinstance(empirical_dist, FrequencyTable):
            raise ValueError(
                "'empirical_dist' argument must be a FrequencyTable class."
            )
        self.e_dist = empirical_dist
        #
        self.alphas = args

    def map(self):
        sum_alpha = sum([(alpha - 1) for alpha in self.alphas])

        def theta_hat(i, level):
            # theta_hat_i = (n_i + \alpha_i - 1)/(N + \sum_j (\alpha_j - 1))
            n_i = self.e_dist.frequency(level, normalised=False)
            alpha_i = self.alphas[i]
            return (n_i + alpha_i - 1) / (self.e_dist.total + sum_alpha)

        return {level: theta_hat(i, level) for i, level in enumerate(self.e_dist)}
