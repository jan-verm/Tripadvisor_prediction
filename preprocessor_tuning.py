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
 from preprocessor_tuning import *
 preprocessor_object = Preprocessor2()
 train_data, test_data = preprocessor_object.load()
 X_train_full, X_val_full, y_train, y_val = preprocessor_object.fit_data(train_data)
 sigma_values = preprocessor_object.compute_sigma_values(X_train_full, y_train, a_value, epsilon)
 X_train, X_val, X_test = preprocessor_object.remove_features(sigma_values, reduction_level, X_train_full, X_val_full, test_data)
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

import data
from utils import dump_pickle, load_pickle

# CONSTANTS
NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.20
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
class Preprocessor2:
    """
    Class for dimensionality reduction and splitting, optimized for hyperparameter tuning.
    Not compatable with postprocessing ensembles.
    """
    def __init__(self):
        """
        Initialization of the class attributes.
        """

        # Initialise stemmed count vectorisation and tfidf transformer
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE)

    def load(self):
        """
        Load the pickeled dataset.

        parameters:
        :return list<Review> dataset['train']: A list of all train (and validation) reviews.
        :return list<Review> dataset['test']: A list of all test reviews.
        """
        log('Loading test and train data...')
        dataset = data.load_pickled_data()
        return dataset['train'], dataset['test']

    def fit_data(self, train_and_val_data):
        """
        Split the trainset in train and validation set. Transform the training set to a tf-idf matrix.

        parameters:
        :param list<Review> train_and_val_data: A list of all train (and validation) reviews.
        :return csr-matrix X_train: The tf-idf feature matrix of the training samples.
        :return list<str> X_val: The validation set reviews' content.
        :return numpy array y_train: The ratings corresponding to the samples in the training feature matrix.
        :return numpy array y_val: The ratings corresponding to the reviews in the validation set.
        """
        X = [review.content for review in train_and_val_data]
        y = np.array([int(review.rating) for review in train_and_val_data])

        # Split into train and validation set (not hotel bases, this was introduced later in the actual preprocessor)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)

        X_train = [X_train[i] for i in range(0,len(X_train)) if y_train[i] != 0]
        y_train = np.array([y_train[i] for i in range(0,len(y_train)) if y_train[i] != 0])

        log('started fitting the TfidfVectorizer for the training data ...')
        # Build features from train set
        X_train = self.tfidf_vectorizer.fit_transform(X_train)

        return X_train, X_val, y_train, y_val

    def compute_sigma_values(self, X_train, y_train, a_value, epsilon):
        """
        Compute the sigma values for all features. This is an indication of their importance.

        parameters:
        :param csr-matrix X_train: The tf-idf feature matrix of the training samples.
        :param numpy array y_train: The ratings corresponding to the samples in the training feature matrix.
        :param int a_value: Hyperparameter used in the dimensionality reduction, to adapt IDF importance.
        :param float epsilon: Hyperparameter used in dimensionality reduction, to fix zero varriance feature importance.
        :return dict<PriorityQueue> sigma_values: a dictionary of all sigma values, seperated by nearest rating for which they are important.
        """
        log('started computing the sigma values for the training data ...')
        # Initialize the sigma priority queues
        sigma_values = {1: PriorityQueue(), 2: PriorityQueue(), 3: PriorityQueue(),
                        4: PriorityQueue(), 5: PriorityQueue()}

        # The number of features is the number of columns in the feature matrix
        all_features = X_train.shape[1]

        # Get IDF values from the transformer
        IDFs = self.tfidf_vectorizer._tfidf._idf_diag.diagonal()

        # Loop over all features

        # Transform matrix to csc to perform more effectively with column operations
        X_train = X_train.tocsc()

        # Loop over all features
        for i in range(0, all_features):
            feature = np.ndarray.flatten(X_train.getcol(i).toarray())

            # Compute sigma value as described in literature
            sigma_set = y_train[feature != 0]
            mean_rating = int(round(sigma_set.mean()))

            # Do not invert this value, as is done in literature, this makes using heapq easier
            sigma_value = (sigma_set.var() + epsilon) * ((IDFs[i]) ** a_value)

            # Store feature index at the rating with nearest mean
            sigma_values[mean_rating].put(i, sigma_value)

        log('Computed all var*idf values!')

        # Transform back into csr format (to delete feature cols)
        X_train = X_train.tocsr()

        return sigma_values

    def remove_features(self, sigma_values, reduction_level, X_train_in, X_val_full):
        """
        Remove the less informative features.

        parameters:
        :param dict<PriorityQueue> sigma_values: a dictionary of all sigma values, seperated by nearest rating for which they are important.
        :param float reduction_level: The fraction of features that should remain after dimensionality reduction.
        :param csr-matrix X_train_in: The tf-idf feature matrix of the training samples.
        :param list<str> X_val_full: A list of the reviews' content from the validation set.
        :return csr-matrix X_train: The tf-idf feature matrix of the training samples (reduced dimensionality).
        :return csr-matrix X_val: The tf-idf feature matrix of the validation samples (reduced dimensionality).
        """

        X_train = X_train_in.copy()
        # The number of features is the number of columns in the feature matrix
        all_features = X_train.shape[1]

        # Keep "reduction_level*100"% of all features
        keep_features = int(math.ceil(float(all_features) * reduction_level))

        # Number of features that may be chosen per rating
        features_per_rating = int(math.ceil(float(keep_features) / 5))

        # Selection of the most informative features
        features_to_keep = []

        log('Removing non informative features from IDF matrix and vocabulary...')

        log('- Selecting features...')
        # Let every rating class choose a feature in a round robin fashion
        for a in range(0, 5 * features_per_rating):
            if not sigma_values[(a % 5) + 1].empty():
                features_to_keep.append(sigma_values[(a % 5) + 1].get())

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

        # save tfidf_values
        stored_idfs = self.tfidf_vectorizer._tfidf._idf_diag.copy()
        
        log('- Removing IDF values...')
        # Remove IDF values that are no longer needed
        self.tfidf_vectorizer._tfidf._idf_diag = drop_cols(self.tfidf_vectorizer._tfidf._idf_diag, remove)
        self.tfidf_vectorizer._tfidf._idf_diag = drop_rows(self.tfidf_vectorizer._tfidf._idf_diag, remove)

        log('Extracting informative features from validation data...')

        # Make a countvectorizer that only uses the words from the reduced vocabulary
        count_vectorizer = CountVectorizer(ngram_range=NGRAM_RANGE, vocabulary=reduced_vocabulary)

        # Extract features from validation set
        X_val = count_vectorizer.transform(X_val_full)
        X_val = self.tfidf_vectorizer._tfidf.transform(X_val)

        #restore idf values
        self.tfidf_vectorizer._tfidf._idf_diag = stored_idfs

        return X_train, X_val


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
    print '[INFO] ' + s


# compute the difference between two lists
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
