from probability.inference import Inferrer


class MultinomialMLEInferrer(Inferrer):
    def __init__(self, empirical_distribution):
        super().__init__(empirical_distribution)
        # Maximum Likelihood estimation
        theta_hats = [self.__ed__.probability(level) for level in self.__ed__.keys()]
        self.thetas = theta_hats

    def probability(self, key):
        indices = [i for i, level in enumerate(self.__ed__.keys()) if level == key]
        if len(indices) == 0:
            return 0

        return self.thetas[indices[0]]
