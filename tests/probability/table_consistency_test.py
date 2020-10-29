from pytest import approx
import numpy as np
from probability.core import Table


def test_statistical_independence_table():
    # P(x,y,z) = P(x)P(y)P(z)
    # to check that, first, create a joint dist. by product
    # then marginalis and multiply again. The final must be equal
    # the joint
    # Note: the multi-variable distributions must be statistically
    #       independent
    s_1 = {
        ("x", 1): 1 * 6,
        ("x", 2): 1 * 4,
        ("y", 1): 9 * 6,
        ("y", 2): 9 * 4,
    }

    table1 = Table(s_1, names=["X1", "X2"])

    s_2 = {
        (1, "high", "under", "x"): 4 * 3 * 1 * 1,
        (1, "high", "normal", "x"): 4 * 3 * 2 * 1,
        (1, "high", "over", "x"): 4 * 3 * 3 * 1,
        (1, "high", "obese", "x"): 4 * 3 * 4 * 1,
        (1, "low", "under", "x"): 4 * 2 * 1 * 1,
        (1, "low", "normal", "x"): 4 * 2 * 2 * 1,
        (1, "low", "over", "x"): 4 * 2 * 3 * 1,
        (1, "low", "obese", "x"): 4 * 2 * 4 * 1,
        (2, "high", "under", "x"): 2 * 3 * 1 * 1,
        (2, "high", "normal", "x"): 2 * 3 * 2 * 1,
        (2, "high", "over", "x"): 2 * 3 * 3 * 1,
        (2, "high", "obese", "x"): 2 * 3 * 4 * 1,
        (2, "low", "under", "x"): 2 * 2 * 1 * 1,
        (2, "low", "normal", "x"): 2 * 2 * 2 * 1,
        (2, "low", "over", "x"): 2 * 2 * 3 * 1,
        (2, "low", "obese", "x"): 2 * 2 * 4 * 1,
        (1, "high", "under", "y"): 4 * 3 * 1 * 3,
        (1, "high", "normal", "y"): 4 * 3 * 2 * 3,
        (1, "high", "over", "y"): 4 * 3 * 3 * 3,
        (1, "high", "obese", "y"): 4 * 3 * 4 * 3,
        (1, "low", "under", "y"): 4 * 2 * 1 * 3,
        (1, "low", "normal", "y"): 4 * 2 * 2 * 3,
        (1, "low", "over", "y"): 4 * 2 * 3 * 3,
        (1, "low", "obese", "y"): 4 * 2 * 4 * 3,
        (2, "high", "under", "y"): 2 * 3 * 1 * 3,
        (2, "high", "normal", "y"): 2 * 3 * 2 * 3,
        (2, "high", "over", "y"): 2 * 3 * 3 * 3,
        (2, "high", "obese", "y"): 2 * 3 * 4 * 3,
        (2, "low", "under", "y"): 2 * 2 * 1 * 3,
        (2, "low", "normal", "y"): 2 * 2 * 2 * 3,
        (2, "low", "over", "y"): 2 * 2 * 3 * 3,
        (2, "low", "obese", "y"): 2 * 2 * 4 * 3,
    }
    table2 = Table(s_2, names=["Y1", "Y2", "Y3", "Y4"])
    single_table3 = Table({11: 2, 22: 4, 33: 3}, names=["Z"])

    joint_dist = table1 * table2 * single_table3

    marginals = []
    for name in joint_dist.names:
        names_except_one = list(set(joint_dist.names) - {name})
        marginal = joint_dist.marginal(*names_except_one)
        marginals.append(marginal)

    joint_dist2 = np.product(marginals)

    # Normalise Both
    t1 = sum(joint_dist.values())
    t2 = sum(joint_dist2.values())

    joint_dist = Table({k: v / t1 for k, v in joint_dist.items()}, joint_dist.names)

    joint_dist2 = Table({k: v / t2 for k, v in joint_dist2.items()}, joint_dist2.names)

    for k1 in joint_dist:
        assert joint_dist[k1] == approx(joint_dist2[k1], abs=1e-16)


def test_factoring_table():
    s_1 = {
        (1, "high", "under", "x"): 4 * 3 * 1 * 1,
        (1, "high", "normal", "x"): 4 * 3 * 2 * 1,
        (1, "high", "over", "x"): 4 * 3 * 3 * 1,
        (1, "high", "obese", "x"): 4 * 3 * 4 * 1,
        (1, "low", "under", "x"): 4 * 2 * 1 * 1,
        (1, "low", "normal", "x"): 4 * 2 * 2 * 1,
        (1, "low", "over", "x"): 4 * 2 * 3 * 1,
        (1, "low", "obese", "x"): 4 * 2 * 4 * 1,
        (2, "high", "under", "x"): 2 * 3 * 1 * 1,
        (2, "high", "normal", "x"): 2 * 3 * 2 * 1,
        (2, "high", "over", "x"): 2 * 3 * 3 * 1,
        (2, "high", "obese", "x"): 2 * 3 * 4 * 1,
        (2, "low", "under", "x"): 2 * 2 * 1 * 1,
        (2, "low", "normal", "x"): 2 * 2 * 2 * 1,
        (2, "low", "over", "x"): 2 * 2 * 3 * 1,
        (2, "low", "obese", "x"): 2 * 2 * 4 * 1,
        (1, "high", "under", "y"): 4 * 3 * 1 * 3,
        (1, "high", "normal", "y"): 4 * 3 * 2 * 3,
        (1, "high", "over", "y"): 4 * 3 * 3 * 3,
        (1, "high", "obese", "y"): 4 * 3 * 4 * 3,
        (1, "low", "under", "y"): 4 * 2 * 1 * 3,
        (1, "low", "normal", "y"): 4 * 2 * 2 * 3,
        (1, "low", "over", "y"): 4 * 2 * 3 * 3,
        (1, "low", "obese", "y"): 4 * 2 * 4 * 3,
        (2, "high", "under", "y"): 2 * 3 * 1 * 3,
        (2, "high", "normal", "y"): 2 * 3 * 2 * 3,
        (2, "high", "over", "y"): 2 * 3 * 3 * 3,
        (2, "high", "obese", "y"): 2 * 3 * 4 * 3,
        (2, "low", "under", "y"): 2 * 2 * 1 * 3,
        (2, "low", "normal", "y"): 2 * 2 * 2 * 3,
        (2, "low", "over", "y"): 2 * 2 * 3 * 3,
        (2, "low", "obese", "y"): 2 * 2 * 4 * 3,
    }

    table1 = Table(s_1, names=["Y1", "Y2", "Y3", "Y4"])

    def normalise(table):
        total = sum(table.values())
        return Table({k: v / total for k, v in table.items()}, table.names)

    # Normalise First
    table1 = normalise(table1)
    #
    table2 = table1.marginal("Y2", "Y3", "Y4")
    table2 = normalise(table2)
    #
    table3 = table1.condition_on("Y1")
    table3 = Table({k: normalise(v) for k, v in table3.items()}, table3.names)
    #
    table4 = table3 * table2
    for k in table1:
        assert table4[k] == approx(table1[k], abs=1e-16)

    table5 = table2 * table3
    for k in table1:
        assert table5[k] == approx(table1[k], abs=1e-16)
    # On two columns
    table6 = table1.marginal("Y3", "Y4")
    table6 = normalise(table6)
    table7 = table1.condition_on("Y1", "Y2")
    table7 = Table({k: normalise(v) for k, v in table7.items()}, table7.names)

    table8 = table6 * table7
    for k in table1:
        assert table8[k] == approx(table1[k], abs=1e-16)

    table9 = table7 * table6
    for k in table1:
        assert table9[k] == approx(table1[k], abs=1e-16)
    # On Three columns
    table10 = table1.marginal("Y4")
    table10 = normalise(table10)
    table11 = table1.condition_on("Y1", "Y2", "Y3")
    table11 = Table({k: normalise(v) for k, v in table11.items()}, table11.names)

    table12 = table10 * table11
    for k in table1:
        assert table12[k] == approx(table1[k], abs=1e-16)

    table13 = table11 * table10
    for k in table1:
        assert table13[k] == approx(table1[k], abs=1e-16)
