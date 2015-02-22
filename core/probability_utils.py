def probability_of_feature(data, index):
    return (
        sum([
            vector[0][index]
            for vector in data
            if vector[index]
        ]) / float(len(data))
    )


def probability_of_class(data, class_label):
    return (
        len([
            vector[1]
            for vector in data
            if vector[1] == class_label
        ]) / float(len(data))
    )
