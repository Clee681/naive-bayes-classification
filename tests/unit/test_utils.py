from sure import expect
from core.matrix_utils import dot

def test_dot_product():
    "Given two vectors, the dot product is calculated directly"
    x = [0, 1, 1]
    y = [1, 2, 3]
    expect(dot(x, y)).to.equal(5)
