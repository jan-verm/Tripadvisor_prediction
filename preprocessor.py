'''
Created on 11-nov.-2016

Required folder structure:
 |-- Data
 |   |-- Test
 |   |-- Train
 |-- Predictions
 |-- utils.py
 |-- data.py
 |-- preprocessor.py

Usage:
 from preprocessor import *
 preprocessor_object = Preprocessor(a-value, epsilon-value, use_cached_features, reduction_level)
 X_train, X_val, y_train, y_val, X_test = preprocessor_object.load_and_preprocess()
 val_rev = preprocessor_object.val_reviews
'''

import os
import csv
import numpy as np
import scipy
import math
import heapq
from time import strftime
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import EnglishStemmer
import langid
import data
from utils import dump_pickle, load_pickle

# CONSTANTS
NGRAM_RANGE = (1, 2)
TEST_SIZE = 0
PREDICTIONS_BASENAME = os.path.join('Predictions', 'prediction')
DEFAULT_DATA_LOCATION = 'Data'
DEFAULT_PICKLE_PATH = os.path.join(DEFAULT_DATA_LOCATION, 'data.pkl')


class PriorityQueue:
    """
    Implementation of a heap (priority is higher for low values).
    """
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


# Class for reading reviews and generating the feature matrices
class Preprocessor:
    """
    Class for dimensionality reduction and splitting.
    """
    def __init__(self, a_value, epsilon, use_cached_features, reduction_level=0.025):
        """
        Initialization of the class attributes.
        
        parameters:
        :param int a_value: Hyperparameter used in the dimensionality reduction, to adapt IDF importance.
        :param float epsilon: Hyperparameter used in dimensionality reduction, to fix zero varriance feature importance.
        :param boolean use_cached_features: Skip preprocessing and load features from pickle files.
        :param flaot reduction_level: The fraction of features that should remain after dimensionality reduction.
        """
        # Set class attributes
        self.a_value = a_value
        self.epsilon = epsilon
        self.use_cached_features = use_cached_features
        self.reduction_level = reduction_level

        # Initialise tf-idf vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE, strip_accents = 'unicode')

        # The textual reviews in the validation set, used for postprocessing later
        self.val_reviews = None

    def load_and_preprocess(self):
        """
        Load data from pickle files and generate feature matrices, or load the cached ones.
        
        parameters:
        :return csr-matrix X_train: The preprocessed tf-idf feature matrix of the training samples.
        :return csr-matrix X_val: The preprocessed tf-idf feature matrix of the validation samples.
        :return numpy array y_train: The ratings corresponding to the samples in the training feature matrix.
        :return numpy array y_val: The ratings corresponding to the samples in the validation feature matrix.
        :return csr-matrix X_test: The preprocessed tf-idf feature matrix of the test samples.
        """

        # Check for pickled data
        if not os.path.exists(DEFAULT_PICKLE_PATH):
            print 'Creating pickle file...'
            data.create_pickled_data(overwrite_old=True)

        # Check for cached features
        if self.use_cached_features:
            try:
                data_tuple = self.load_all()
                feature_count = data_tuple[0].shape[1]
                log('Loaded the cached preprocessed data. ({} features)'.format(feature_count))
                self.val_reviews = self.load_val_reviews()
                return data_tuple
            except IOError:
                self.use_cached_features = False
                log('Could not load cached preprocessed data! Doing the preprocessing now...')

        # Preprocess data
        if not self.use_cached_features:
            log('Loading test and train data...')
            dataset = data.load_pickled_data()

            log('Extracting features and reducing the dimensionality...')
            X_train, X_val, y_train, y_val, X_test = self.transform_data(dataset['train'], dataset['test'])

            # Save preprocessed data for later
            self.dump_all(X_train, X_val, y_train, y_val, X_test)
            self.dump_val_reviews(self.val_reviews)

        return X_train, X_val, y_train, y_val, X_test

    def train_test_split_by_hotels(self, data, test_size):
        """
        Build a validation set, which only contains reviews of hotels that are not in the training set.

        parameters:
        :param list<Review> data: A list of all reviews (training and validation).
        :param float test_size: The fraction of all reviews, that should be used for validation.
        :return list<Review> X_train: A list of all training reviews' content.
        :return list<Review> X_val: A list of all validation reviews' content.
        :return numpy array y_train: A list of all training reviews' ratings (corresponding to X_train).
        :return numpy array y_val: a list of all validation reviews! ratings (corresponding to X_val).
        """
        
        # Determine number of validations reviews
        test_limit = len(data) * test_size

        # Create hotel dictionary to count reviews per hotel
        hotel_dict = {}
        for review in data:
            hotel = review.hotel.id
            if hotel not in hotel_dict:
                hotel_dict[hotel] = 0
            hotel_dict[hotel] += 1

        # Pick hotels until the test size is reached
        review_count_buffer = 0
        test_hotels = []
        while review_count_buffer < test_limit:
            picked_hotel = np.random.choice(hotel_dict.keys())
            test_hotels.append(picked_hotel)  # save the list of hotel ids
            review_count_buffer += hotel_dict[picked_hotel]  # add review count
            del hotel_dict[picked_hotel]  # delete hotel from dictionary

        # Create the four lists
        X_train = []
        X_val = []
        y_train = []
        y_val = []

        for review in data:
            hotel = review.hotel.id
            if hotel in test_hotels:  # review belongs to validation set
                X_val.append(review.content)
                y_val.append(review.rating)
            else:  # review belongs to training set
                if review.rating != 0:  # only append to training data if it's not a zero rating review
                    X_train.append(review.content)
                    y_train.append(review.rating)
        
        # Bind the validation reviews to the preprocessor object
        self.val_reviews = X_val
        return X_train, X_val, np.array(y_train), np.array(y_val)

    def transform_data(self, train_and_val_data, test_data):
        """
        Performing the dimensionality reduction.

        parameters:
        :param list<Review> train_and_val_data: A list of all reviews (training and validation).
        :param list<Review> test_data: A list of the test reviews.
        :return csr-matrix X_train: The tf-idf feature matrix of the training samples (reduced dimensionality).
        :return csr-matrix X_val: The tf-idf feature matrix of the validation samples (reduced dimensionality).
        :return numpy array y_train: The ratings corresponding to the samples in the training feature matrix.
        :return numpy array y_val: The ratings corresponding to the samples in the validation feature matrix.
        :return csr-matrix X_test: The preprocessed tf-idf feature matrix of the test samples.
        """

        # Split into train and validation set
        X_train, X_val, y_train, y_val = self.train_test_split_by_hotels(train_and_val_data, test_size=TEST_SIZE)

        # Build features from train set
        X_train = self.tfidf_vectorizer.fit_transform(X_train)

        # Start of the selection of the most useful features

        # Initialize the sigma priority queues
        sigma_values = {1: PriorityQueue(), 2: PriorityQueue(), 3: PriorityQueue(),
                        4: PriorityQueue(), 5: PriorityQueue()}
        number_of_classes = len(sigma_values)

        # The number of features is the number of columns in the feature matrix
        all_features = X_train.shape[1]

        # Keep "reduction_level*100"% of all features
        keep_features = int(math.ceil(float(all_features) * self.reduction_level))

        # Number of features that may be chosen per rating
        features_per_rating = int(math.ceil(float(keep_features) / number_of_classes))

        # Get IDF values from the transformer
        IDFs = self.tfidf_vectorizer._tfidf._idf_diag.diagonal()

        # Loop over all features

        # Use a counter to visualize the progress
        count = 1.0
        prev = 0.0

        # Transform matrix to csc to perform more effectively with column operations
        X_train = X_train.tocsc()

        # Loop over all features
        for i in range(0, all_features):
            feature = np.ndarray.flatten(X_train.getcol(i).toarray())

            # Compute sigma value as described in literature
            sigma_set = y_train[feature != 0]
            mean_rating = int(round(sigma_set.mean()))

            # Do not invert this value, as is done in literature, this makes using heapq easier
            sigma_value = (sigma_set.var() + self.epsilon) * ((IDFs[i]) ** self.a_value)

            # Store feature index at the rating with nearest mean
            sigma_values[mean_rating].put(i, sigma_value)

            # Print "counter" for visualizing progress
            if count / all_features > prev:
                print strftime("%H:%M:%S") + ' : ' + str(
                    math.ceil(100 * (count / all_features * 100)) / 100) + '% of the features checked for sigma values.'
                prev += 0.01
            count += 1.0

        log('Computed all var*idf values!')

        # Selection of the most informative features
        features_to_keep = []

        # Transform back into csr format (to delete feature cols)
        X_train = X_train.tocsr()

        log('Removing non informative features from IDF matrix and vocabulary...')

        log('- Selecting features...')
        # Let every rating class choose a feature in a round robin fashion
        for a in range(0, number_of_classes * features_per_rating):
            if not sigma_values[(a % number_of_classes) + (6 - number_of_classes)].empty():
                features_to_keep.append(sigma_values[(a % number_of_classes) + (6 - number_of_classes)].get())

        log('- Removing features...')
        # Remove all features that will not be used
        log('    - list_diff')
        remove = list_diff(range(0, all_features), features_to_keep)
        log('    - drop_cols')
        X_train = drop_cols(X_train, remove)

        log('- Removing words from the vocabulary...')
        # Remove all words from the vocabulary that won't be used, adept indices
        features_to_keep = set(features_to_keep)
        reduced_vocabulary_temp = {k: v for (k, v) in self.tfidf_vectorizer.vocabulary_.items() if
                                   v in features_to_keep}
        sorted_voc = sorted(reduced_vocabulary_temp.items(), key=lambda x: x[1])
        reduced_vocabulary = {}
        for z in range(0, len(reduced_vocabulary_temp)):
            reduced_vocabulary[sorted_voc[z][0]] = z

        log('- Removing IDF values...')
        # Remove IDF values that are no longer needed
        self.tfidf_vectorizer._tfidf._idf_diag = drop_cols(self.tfidf_vectorizer._tfidf._idf_diag, remove)
        self.tfidf_vectorizer._tfidf._idf_diag = drop_rows(self.tfidf_vectorizer._tfidf._idf_diag, remove)

        log('Extracting informative features from validation data...')

        # Make a countvectorizer that only uses the words from the reduced vocabulary
        count_vectorizer = CountVectorizer(ngram_range=NGRAM_RANGE, vocabulary=reduced_vocabulary, strip_accents = 'unicode')

        if TEST_SIZE > 0:
            # Extract features from validation set
            X_val = count_vectorizer.transform(X_val)
            X_val = self.tfidf_vectorizer._tfidf.transform(X_val)

        log('Extracting informative features from test data...')

        # Extract features from test set
        X_test = count_vectorizer.transform([review.content for review in test_data])
        X_test = self.tfidf_vectorizer._tfidf.transform(X_test)

        return X_train, X_val, y_train, y_val, X_test

    def dump_all(self, X_train, X_val, y_train, y_val, X_test):
        """
        Dump the preprocessed data, for later use of cached features.

        parameters:
        :param csr-matrix X_train: The tf-idf feature matrix of the training samples (reduced dimensionality).
        :param csr-matrix X_val: The tf-idf feature matrix of the validation samples (reduced dimensionality).
        :param numpy array y_train: The ratings corresponding to the samples in the training feature matrix.
        :param numpy array y_val: The ratings corresponding to the samples in the validation feature matrix.
        :param csr-matrix X_test: The preprocessed tf-idf feature matrix of the test samples.
        """
        dump_pickle(data=X_train, path=os.path.join(DEFAULT_DATA_LOCATION, 'X_train.pkl'))
        dump_pickle(data=X_val, path=os.path.join(DEFAULT_DATA_LOCATION, 'X_val.pkl'))
        dump_pickle(data=y_train, path=os.path.join(DEFAULT_DATA_LOCATION, 'y_train.pkl'))
        dump_pickle(data=y_val, path=os.path.join(DEFAULT_DATA_LOCATION, 'y_val.pkl'))
        dump_pickle(data=X_test, path=os.path.join(DEFAULT_DATA_LOCATION, 'X_test.pkl'))

    # Load all feature sets
    def load_all(self):
        """
        Load the preprocessed data.

        parameters:
        :return csr-matrix X_train: The tf-idf feature matrix of the training samples (reduced dimensionality).
        :return csr-matrix X_val: The tf-idf feature matrix of the validation samples (reduced dimensionality).
        :return numpy array y_train: The ratings corresponding to the samples in the training feature matrix.
        :return numpy array y_val: The ratings corresponding to the samples in the validation feature matrix.
        :return csr-matrix X_test: The preprocessed tf-idf feature matrix of the test samples.
        """
        X_train = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'X_train.pkl'))
        X_val = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'X_val.pkl'))
        y_train = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'y_train.pkl'))
        y_val = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'y_val.pkl'))
        X_test = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'X_test.pkl'))
        return X_train, X_val, y_train, y_val, X_test

    def load_val_reviews(self):
        """
        Load the list of reviews' content from the validation set.

        parameters:
        :return list<str> val_rev: A list of the reviews' content of the validation set.
        """
        val_rev = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'reviews_val.pkl'))
        return val_rev

    def dump_val_reviews(self, val_rev):
        """
        Dump the list of reviews' content from the validation set.

        parameters:
        :param list<str> val_rev: A list of the reviews' content of the validation set.
        """
        dump_pickle(data=val_rev, path=os.path.join(DEFAULT_DATA_LOCATION, 'reviews_val.pkl'))

    # Helper function used by tfidf_vectorizer in transform_data(...)
    def stemmed_words(self, doc):
        """
        Used for stemming the review content (no longer used).

        parameters:
        :param str doc: The content of a review.
        :return str (): The stemmed review content.
        """
        return (self.stemmer.stem(w) for w in self.analyzer(doc))


# Logging function
def log(s):
    print '[INFO] ' + str(s)


# Compute the difference between two lists
def list_diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


# Helper functions to manipulate sparse matrices
def drop_cols(M, idx_to_drop):
    """
	Remove columns from (a sparse csr) matrix.
	When working with sparse matrices, no deep copies are made.
	imput arguments:
		M: the matrix from which to remove columns
		idx_to_drop: the indices of the columns to remove
	output arguments:
		C.tocsr: csr matrix, with only the remaining columns
	"""

    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)  # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()


def drop_rows(M, idx_to_drop):
    """
	Remove rows from a sparse csr matrix.
	When working with sparse matrices, no deep copies are made.
	imput arguments:
		M: the matrix from which to remove rows
		idx_to_drop: the indices of the rows to remove
	output arguments:
		M[mask]: csr matrix, with only the remaining rows
	"""
    if not isinstance(M, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(idx_to_drop)
    mask = np.ones(M.shape[0], dtype=bool)
    mask[idx_to_drop] = False
    return M[mask]
