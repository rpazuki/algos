import numpy as np
from probability.core_2 import Table


def multinomial_map(data_table, alphas=None):
    """multinomial parameter's maximum a posteriori estimate.

        Prior:                  Dirichlet(mu|alpha)
        likelihood:             P(D|mu) ~ Pi_i^m mu_i^{x_i}
        Posterior:              P(mu| D) ~ P(D|mu)P(mu|alpha)
                                         ~ Dirichlet(mu|alpha+D)
        (multinulli)
        MAP:   mu_i = (alpha_i + x_i -1)/(sum_i^m alpha_i + n - m)

    Args:
        data_table (Table): [description]
        alphas ([type], optional): [description]. Defaults to None.
    """
    if type(data_table) != Table:
        raise ValueError("'data_table' argument must be a 'Table' class.")

    if alphas is not None and len(alphas) != len(data_table):
        raise ValueError("'alphas' length is not equal to 'data_table'.")

    if alphas is None:
        alphas = np.ones(len(data_table))

    values = alphas + data_table.np_values() - 1
    values = values / np.sum(values)
    return Table(
        {k: v for k, v in zip(data_table.keys(), values)},
        data_table.names,
        _internal_=True,
    )


def multinulli(data_table, alphas=None):
    """Multinulli (single event) posterior predictive.

        Prior:                  Dirichlet(mu|alpha)
        likelihood:             P(D|mu) ~ Pi_i^m mu_i^{x_i}
        Posterior:              P(mu| D) ~ P(D|mu)P(mu|alpha)
                                         ~ Dirichlet(mu|alpha+D)
        (multinulli)
        Posterior predictive:   p(X_j = j| D) = (alpha_j+x_j)/(alpha_0+N)

    Args:
        data_table (Table): [description]
        alphas ([type], optional): [description]. Defaults to None.
    """
    if type(data_table) != Table:
        raise ValueError("'data_table' argument must be a 'Table' class.")

    if alphas is not None and len(alphas) != len(data_table):
        raise ValueError("'alphas' length is not equal to 'data_table'.")

    if alphas is None:
        alphas = np.ones(len(data_table))

    values = alphas + data_table.np_values()
    values = values / np.sum(values)
    return Table(
        {k: v for k, v in zip(data_table.keys(), values)},
        data_table.names,
        _internal_=True,
    )


class Multinomial(Table):
    def __init__(self, data_table, alphas=None):
        """multinomial parameter's maximum a posteriori estimate.

        Prior:                  Dirichlet(mu|alpha)
        likelihood:             P(D|mu) ~ Pi_i^m mu_i^{x_i}
        Posterior:              P(mu| D) ~ P(D|mu)P(mu|alpha)
                                         ~ Dirichlet(mu|alpha+D)
        (multinulli)
        Posterior predictive:
             P(X_j = j| D) = (alpha_j+x_j)/(sum_i (alpha_i + x_i) )

        Args:
            data_table (Table): [description]
            alphas ([type], optional): [description]. Defaults to None.
        """
        if type(data_table) != Table:
            raise ValueError("'data_table' argument must be a 'Table' class.")

        if alphas is not None and len(alphas) != len(data_table):
            raise ValueError("'alphas' length is not equal to 'data_table'.")

        if alphas is None:
            alphas = np.ones(len(data_table))
        else:
            alphas = np.array(alphas)

        self.alphas = alphas
        values = alphas + data_table.np_values()
        values = values / np.sum(values)
        super().__init__(
            dict(zip(data_table.keys(), values)),
            data_table.names,
            _internal_=True,
        )
