from sure import expect
from core.matrix_utils import dot
from core.probability_utils import (
    probability_of_feature,
    probability_given,
    probability_of_class)

def test_dot_product():
    "Given two vectors, the dot product is calculated correctly"
    x = [0, 1, 1]
    y = [1, 2, 3]
    expect(dot(x, y)).to.equal(5)


def test_probability_of_feature():
    "Given list of lists, the probability of feature1 is calculated"
    "['feature1', 'feature2']"
    data = [
        [[1, 0], True],
        [[0, 0], False]
    ]
    expect(probability_of_feature(data, 0)).to.equal(0.5)
    expect(probability_of_feature(data, 1)).to.equal(0.0)


def test_probability_of_success():
    """
    Given the training data, compute the probability of the
    positive class.
    """
    data = [
        [[1, 0], True],
        [[0, 0], False],
        [[0, 0], False],
        [[0, 1], False]
    ]
    expect(probability_of_class(data, True)).to.equal(0.25)
    expect(probability_of_class(data, False)).to.equal(0.75)

def test_probability_given():
    "Given a class label, compute the probability of feature x"
    data = [
        [[1, 0], True],
        [[0, 0], False],
        [[0, 0], False],
        [[0, 1], False]
    ]
    expect(probability_given(data, True, 0)).to.equal(1.0)
    expect(probability_given(data, True, 1)).to.equal(0.0)
    expect(probability_given(data, False, 1)).to.equal(1/3.0)
    expect(probability_given(data, False, 0)).to.equal(0.0)
