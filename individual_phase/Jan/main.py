# -*- coding: utf-8 -*-

'''
Created on 25-okt.-2016

Required folder structure:
 |-- Data
 |   |-- Test
 |   |-- Train
 |-- Predictions 
 |-- utils.py
 |-- data.py
 |-- main.py

@author: Jan Vermeulen
'''

import os
import csv
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from nltk.stem.snowball import EnglishStemmer
#from sklearn.feature_selection import RFE
#from sklearn.decomposition import TruncatedSVD

import data
import utils
from utils import dump_pickle, load_pickle

NUM_THEADS = 4
NGRAM_RANGE = (1, 3)
TEST_SIZE = 0.01
PREDICTIONS_BASENAME = os.path.join('Predictions', 'prediction')
DEFAULT_DATA_LOCATION = 'Data'
DEFAULT_PICKLE_PATH = os.path.join(DEFAULT_DATA_LOCATION, 'data.pkl')

stemmer = EnglishStemmer()
analyzer = CountVectorizer(analyzer='word').build_analyzer()

USE_CACHED_FEATURES = True

def main():
    if not os.path.exists(DEFAULT_PICKLE_PATH):
        print 'Creating pickle file...'
        data.create_pickled_data(overwrite_old=True)
        
    if not USE_CACHED_FEATURES:
        log('Loading test and train data...')
        dataset = data.load_pickled_data()
        
        log('Extracting features and target...')
        X_train, X_val, y_train, y_val, X_test = transform_data(dataset['train'], dataset['test'])
        
        print 'train feature shape: %d' % X_train.shape[1]
        print 'val feature shape: %d' % X_val.shape[1]
        
        dump_all(X_train, X_val, y_train, y_val, X_test)
    else:
        X_train, X_val, y_train, y_val, X_test = load_all()
    
    # Using TruncatedSVD
    #tsvd = TruncatedSVD(n_components=5000)
    #X_train = tsvd.fit_transform(X_train)
    #X_val = tsvd.fit_transform(X_val)
    #X_test = tsvd.fit_transform(X_test)
    
    log('Training model...')
    model = LogisticRegression(solver='sag', n_jobs=NUM_THEADS, C=5, tol=0.01)
    # Using RFE
    #model = RFE(model, n_features_to_select=80000, step=10000, verbose=1)
    model = model.fit(X_train, y_train)
    
    log('Predicting train and validation set...')
    predictions_train = model.predict(X_train)
    log('Train error = %s' % str(mean_absolute_error(predictions_train, y_train)))
    predictions_val = model.predict(X_val)
    log('Validation error = %s' % str(mean_absolute_error(predictions_val, y_val)))
    
    log('Predicting test set...')
    predictions_test = model.predict(X_test)
    
    pred_file_name = utils.generate_unqiue_file_name(PREDICTIONS_BASENAME, 'csv')
    log('Dumping predictions to %s...' % pred_file_name)
    write_predictions_to_csv(predictions_test, pred_file_name)
    
    log('That\'s all folks!')


def transform_data(train_and_val_data, test_data):
    # Initialise stemmed count vectorisation and tfid transformer
    count_vectorizer = CountVectorizer(ngram_range=NGRAM_RANGE, stop_words='english', analyzer=stemmed_words)
    tfid_transformer = TfidfTransformer(use_idf=True)
    
    X = [review.content for review in train_and_val_data]
    y = np.array([review.rating for review in train_and_val_data])
    
    # Split into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)
    
    # Build features from train set
    X_train = count_vectorizer.fit_transform(X_train)    
    X_train = tfid_transformer.fit_transform(X_train)
    
    # Extract features from validation set
    X_val = count_vectorizer.transform(X_val)
    X_val = tfid_transformer.transform(X_val)
    
    # Extract features from test set
    X_test = count_vectorizer.transform([review.content for review in test_data])
    X_test = tfid_transformer.transform(X_test)
    
    return X_train, X_val, y_train, y_val, X_test


# Dump all feature sets
def dump_all(X_train, X_val, y_train, y_val, X_test):
    dump_pickle(data = X_train, path=os.path.join(DEFAULT_DATA_LOCATION, 'X_train.pkl'))
    dump_pickle(data = X_val, path=os.path.join(DEFAULT_DATA_LOCATION, 'X_val.pkl'))
    dump_pickle(data = y_train, path=os.path.join(DEFAULT_DATA_LOCATION, 'y_train.pkl'))
    dump_pickle(data = y_val, path=os.path.join(DEFAULT_DATA_LOCATION, 'y_val.pkl'))
    dump_pickle(data = X_test, path=os.path.join(DEFAULT_DATA_LOCATION, 'X_test.pkl'))
    
    
# Load all feature sets
def load_all():
    X_train = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'X_train.pkl'))
    X_val = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'X_val.pkl'))
    y_train = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'y_train.pkl'))
    y_val = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'y_val.pkl'))
    X_test = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'X_test.pkl'))
    return X_train, X_val, y_train, y_val, X_test


# Helper function used by count_vectorizer in transform_data(...)
def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


# Logging function
def log(s):
    print '[INFO] ' + s
   
   
# Output helper function - Copied from example files
def write_predictions_to_csv(predictions, out_path):
    """Writes the predictions to a csv file.
    Assumes the predictions are ordered by review id.
    """
    with open(out_path, 'wb') as outfile:
        # Initialise the writer
        csvwriter = csv.writer(
            outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Write the header
        csvwriter.writerow(['id', 'rating'])
        # Write the rows using 18 digit precision
        for idx, prediction in enumerate(predictions):
            csvwriter.writerow([str(idx + 1), "%.18f" % prediction])
            
            
if __name__ == '__main__':
    main()
