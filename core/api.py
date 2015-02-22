from copy import copy
import random
import csv
from .probability_utils import compute_training_weights
from .matrix_utils import dot

THRESHOLD = 0.8

class TwitterClassifier(object):
    source_filename = 'data/clean_twitter_data.csv'

    def __init__(self):
        with open(self.source_filename, 'rb') as f:
            self.data = [row for row in csv.reader(f)]

    def get_checks(self):
        """
        returns the boolean checks we want to do on the text
        to come up with the feature vector
        """
        return [
            lambda text: "bad" in text.lower(),
        ]

    def get_feature_vector(self, row):
        """
        Transform twitter text into a feature vector of 0s and 1s, e.g.

        "I bend backwards" -> [0 0 0 0 1 1 0 1]
        """
        sentiment = bool(row[0])
        text = row[1]
        feature_vector = [
            check(text)
            for check in self.get_checks()
        ]

        return [
            feature_vector,
            sentiment
        ]


    def get_data_sets(self, data, ratio=.2):
        """
        return training_set, test_set

        the ratio is of the size of one set to the other
        """
        data = copy(data)
        random.shuffle(data)
        endpoint = int(.2 * len(data))
        return data[0: endpoint], data[endpoint: ]


    def train(self, data):
        """
        Based on `data`, a list of feature vectors coupled with the class_label, e.g.
        data = [
            [(1,0,0,1), True],
            ....
        ]

        compute the probability weights,
        and return the weights vector, a.k.a the NBC model.
        """
        return compute_training_weights(data, True)

    def get_accuracy(self, nbc_model, test_data):
        """
        Return accuracy based on the test_data, which is in 0s and 1s feature vector form.
        """
        total_count = len(test_data)
        correct = 0

        for feature_vector, answer in test_data:
            # compute the probability from the bayesian formula
            dot_product = dot(nbc_model, feature_vector)

            # if we guessed positive but it was negative or vice versa,
            # then we're wrong
            wrong = (
                (answer and dot_product < THRESHOLD) or
                (not answer and dot_product > THRESHOLD))

            if not wrong:
                correct += 1
        return correct / float(total_count)

    def process(self):
        """
        1. Read in data.
        2. Translate into feature vectors
        3. Separate into training and test sets
        4. Train the NBC model
        5. Test the accuracy on the test set
        """
        vectors = [self.get_feature_vector(row) for row in self.data]
        training_set, test_set = self.get_data_sets(vectors)
        nbc_model = self.train(training_set)
        accuracy = self.get_accuracy(nbc_model, test_set)
        print "Accuracy :{}".format(accuracy)


if __name__ == "__main__":
    TwitterClassifier().process()
