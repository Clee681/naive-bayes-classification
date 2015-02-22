def probability_of_feature(data, index):
    return (
        sum([vector[index] for vector in data if vector[index]]) / float(len(data))
    )
