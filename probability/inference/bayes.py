import numpy as np

# from scipy.stats import dirichlet
from probability.core_2 import Table


class Multinomial(Table):
    def __init__(self, data_table, alphas=None):
        if type(data_table) != Table:
            raise ValueError("'data_table' argument must be a 'Table' class.")

        if alphas is None:
            self.alphas = np.ones(len(data_table))
        else:
            self.alphas = np.array(alphas)
        # Bishop p. 77
        # Posterior P(mu | D, alpha) ~ Dir(mu | m + alpha)
        updated_alphas = self.alphas + data_table.np_values()
        super().__init__(
            {k: v for k, v in zip(data_table.keys(), updated_alphas)},
            data_table.names,
            _internal_=True,
        )

    def __getitem__(self, args):
        """Use predictive distribution to find the probability
        P(X=x_i| D) = \\int_0^1 P(X=x_i|mu) P(mu| D) dmu
                    = \\int_0^1 P(X=x_i|mu) Dir(mu | m + alpha) dmu
                    = \\int_0^1 mu_k Dir(mu | m + alpha) dmu
                    = E[mu_k | D]
        """
        alpha = super().__getitem__(args)
        if alpha is None:
            return 0
        total = self.get_total()
        if total == 0:
            return 0

        return alpha / total
