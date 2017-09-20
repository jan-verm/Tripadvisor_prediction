import data
import operator
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from preprocessor import log
import re


def non_ascii(dataset):
    """
    Plot a histogram of reviews containing more than 90% non-ascii chars
    (or less than 10% ascii chars)
    """
    distr_per_class = [0, 0, 0, 0, 0, 0]
    data_per_class = [0, 0, 0, 0, 0, 0]
    data_per_class_ascii = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        data_per_class[i] = [review.content for review in dataset if review.rating == i]
        data_per_class_ascii[i] = [review for review in data_per_class[i] if len(review) > 0 and float(len(re.sub('[ -~]', '', review)))/float(len(review)) > 0.9 ]
        distr_per_class[i] = float(len(data_per_class_ascii[i]))/float(len(dataset))
    plt.bar(range(6), distr_per_class)
    plt.xlabel('rating')
    plt.ylabel('fraction of reviews')
    plt.show()


def histogram_ratings(data):
    """
    Plot a histogram of the ratings in the given data.
    """
    ratings = [review.rating for review in data]
    plt.hist(ratings, bins=6, normed=True)
    plt.title("Ratings histogram")
    plt.show()


def dataset_info(train_data, test_data):
    log("Training size: {}, test size: {}".format(len(train_data), len(test_data)))


def list_authors(train_data, test_data):
    """
    Print some information about the authors in the train_authors and test data.
    """
    train_authors = [x.author for x in train_data]
    test_authors = [x.author for x in test_data]
    both = train_authors + test_authors

    author_counter = Counter(both)
    log('Most common authors in both: {}'.format(author_counter.most_common(3)))

    test_unique = list(set(test_authors) - set(train_authors))
    test_authors_percentage = get_percentage(len(test_unique), len(set(test_authors)))
    log('Unique authors in test_data: {} ({}%)'.format(len(test_unique), test_authors_percentage))

    ratings = np.array([x.rating for x in train_data])
    train_authors = np.array(train_authors)
    zero_authors = train_authors[ratings == 0]
    log("Authors with zero ratings: (length:{}) {}".format(len(zero_authors), set(zero_authors)))

    test_authors = np.array(test_authors)
    for author_tuple in author_counter.most_common(2):
        author = author_tuple[0]
        author_percentage = get_percentage(author_tuple[1], len(test_data) + len(train_data))
        log("Amount of test reviews by {}: {} ({}%)".format(author, len(test_authors[test_authors == author]), author_percentage))


def get_percentage(amount, total):
    return round(float(amount) / total * 100, 1)


def histogram_amount_of_reviews_per_hotel(train_data):
    dict = {}
    zero_ratings = {}
    for review in train_data:
        hotel = review.hotel.id
        if review.rating == 0:
            if hotel not in zero_ratings:
                zero_ratings[hotel] = 0
            zero_ratings[hotel] += 1
        if hotel not in dict:
            dict[hotel] = 0
        dict[hotel] += 1

    for (hotel, count) in zero_ratings.iteritems():
        zero_ratings[hotel] = get_percentage(count, dict[hotel])

    print "Hotels with zero ratings: {}".format(get_percentage(len(zero_ratings), len(dict.keys())))
    print sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    print sorted(zero_ratings.items(), key=operator.itemgetter(1), reverse=True)


def main():
    data_set = data.load_pickled_data()
    train_data = data_set['train']
    test_data = data_set['test']

    non_ascii(train_data)
    exit()
    # dataset_info(train_data, test_data)
    # list_authors(train_data, test_data)
    histogram_ratings(train_data)
    histogram_amount_of_reviews_per_hotel(train_data)

if __name__ == '__main__':
    main()
