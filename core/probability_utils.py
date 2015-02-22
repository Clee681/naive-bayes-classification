def probability_of_feature(data, index):
    return (
        sum([
            vector[0][index]
            for vector in data
            if vector[0][index]
        ]) / float(len(data))
    )


def filter_on_class_label(data, class_label):
    return [
        vector
        for vector in data
        if vector[1] == class_label
    ]


def probability_of_class(data, class_label):
    return (
        len(filter_on_class_label(data, class_label)) /
        float(len(data))
    )

def probability_given(data, class_label, feature_index):
    """
    Compute P(feature | class_label)
    """
    class_vectors = filter_on_class_label(data, class_label)
    return probability_of_feature(class_vectors, feature_index)


def reverse_probability_given(data, class_label, feature_index):
    """
    Compute P(class_label | feature)
    """
    return (
        probability_given(data, class_label, feature_index) *
        probability_of_class(data, class_label)
    ) / (probability_of_feature(data, feature_index))


def compute_training_weights(data, class_label):
    feature_count = len(data[0][0])
    return [
        reverse_probability_given(data, class_label, i)
        for i in range(feature_count)
    ]
