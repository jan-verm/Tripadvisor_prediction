import data
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from pipeline import learn


def histogram_ratings(data):
    """
    Plot a histogram of the ratings in the given data.
    """
    ratings = [review.rating for review in data]
    plt.hist(ratings, bins=6, normed=True)
    plt.title("Ratings histogram")
    plt.show()


def histogram_prices(data):
    """
    Plot a histogram of the prices in the given data.
    """
    prices = [x.hotel.price for x in data]
    plt.hist(prices, bins=20, normed=True)
    plt.title("Prices histogram")
    plt.show()


def list_authors(train_data, test_data):
    """
    Print some information about the authors in the train and test data.
    """
    train = [x.author for x in train_data]
    test = [x.author for x in test_data]
    both = train + test

    counter = Counter(both)
    print '\nMost common authors in both: {}'.format(counter.most_common(2))

    test_unique = list(set(test) - set(train))
    print 'Unique authors in test_data: {}'.format(len(test_unique))


def parameter_refinement():
    """
    Try several parameter configurations to determine an approximation of the optimal solution.
    """
    print "LINEAR SVC:"
    print "c,mae"
    for c in np.arange(0.1, 2.01, 0.05):
        mae = learn(c)
        print "{},{}".format(c,mae)

    print "\nMULTINOMIAL NB:"
    print "a,mae"
    for a in np.arange(0.0125, 0.018, 0.0005):
        mae = learn(a, 'mnb')
        print "{},{}".format(a, mae)


def main():
    data_set = data.load_pickled_data()
    train_data = data_set['train']
    test_data = data_set['test']

    histogram_ratings(train_data)
    histogram_prices(train_data + test_data)
    list_authors(train_data, test_data)


if __name__ == '__main__':
    main()
