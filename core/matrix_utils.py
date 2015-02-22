def dot(x, y):
    """
    Return the vector dot product of x and y.

    x and y are vectors, necessarily of the same length, represented by lists.
    """
    return sum([i * j for i, j in zip(x, y)])
