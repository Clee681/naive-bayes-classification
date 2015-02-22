from sure import expect
from core.matrix_utils import dot
from core.probability_utils import probability_of_feature

def test_dot_product():
    "Given two vectors, the dot product is calculated directly"
    x = [0, 1, 1]
    y = [1, 2, 3]
    expect(dot(x, y)).to.equal(5)

def test_probability_of_feature():
    "Given list of lists, the probability of feature1 is calculated"
    "['feature1', 'feature2']"
    data = [[1,0],[0,0]]
    expect(probability_of_feature(data, 0)).to.equal(0.5)
