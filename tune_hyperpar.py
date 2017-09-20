'''
Created on 14-nov.-2016

Required folder structure:
 |-- Data
 |   |-- Test
 |   |-- Train
 |-- Predictions
 |-- utils.py
 |-- data.py
 |-- preprocessor.py
 |-- tune_hyperpar.py

usage: set MAIN_TO_RUN correctly
run the following command: python tune_hyperpar.py <args>
all main args:
start_a_value: a-value from which to start the search range
stop_a_value: a-value at which to stop the search range
single model args:
model_name: The name of the model

'''

from preprocessor import *
from preprocessor_tuning import *
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn import svm
from sklearn.metrics import mean_absolute_error
from mord import OrdinalRidge, LAD
from sklearn.svm.classes import LinearSVC
from sklearn.model_selection import ParameterGrid, GridSearchCV
from multiprocessing.pool import ThreadPool
from sklearn.naive_bayes import MultinomialNB
import cPickle as pickle
import sys
import copy

from time import strftime

import utils
import data

# CONSTANTS

NUM_THEADS=4
MAIN_TO_RUN = 'single'
DEFAULT_DATA_LOCATION = 'Data'
NUM_FOLDS = 4

# possible values of MAIN_TO_RUN: 
# 'pre' for preprocessor hyperparameter tuning; 
# 'all': for both model and preprocessor hyperparameter tuning:
# 'mod': for only model hyperparameter tuning (using best preprocessor hyperparameters)
# 'single': for only model (not the ensemble) hyperparameter tuning (using best preprocessor hyperparameters)

BEST_A = 11
BEST_EPSILON = 0.1
BEST_RED_LVL = 0.025

def main_all():
    """
    The main method that can be used for an exhaustive search for the best hyperparameter combination,
    among both preprocessor hyperparameters and model hyperparameters.
    Uncomment the ensemble/method that should be tested.
    """
    # Initialize the result array
    results = []

    # Set the search range in a parameter grid, all possible combinations will be tested
    params_preprocessor = {'a_value':range(int(sys.argv[1]), int(sys.argv[2])), 'epsilon':[0.1], 'reduction_level':[0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05]}
    #params_grid = ParameterGrid({'logistic__C': [0.2,0.5,0.7], 'logistic__tol':[0.00001,0.0001,0.001], 'svc__C': [0.2,0.5,0.7], 'svc__tol':[0.00001,0.0001,0.001],'ridge__alpha':[0.01,0.1,1,10,100]})
    params_grid = ParameterGrid({'logistic__C': [5], 'logistic__tol':[0.001]}) #{'logistic__C': np.logspace(-5.0, 5.0, num=5), 'logistic__tol':np.logspace(-5.0, 5.0, num=5)}
    
    # Keep the best score in order to only print improvements
    best = 1
    
    # Create a preprocessor object (tweaked for tuning hyperparameters)
    preprocessor_object = Preprocessor2()
    
    # load the data
    train_data, test_data = preprocessor_object.load()
    
    # Convert the data to tf-idf matrices
    X_train_full, X_val_full, y_train, y_val = preprocessor_object.fit_data(train_data)
    
    # Test the preprocessor and the model for all possible hyperparameter combinations
    for x in params_preprocessor['a_value']:
        for y in params_preprocessor['epsilon']:
            sigma_values = preprocessor_object.compute_sigma_values(X_train_full, y_train, x, y)
            for z in params_preprocessor['reduction_level']:
                log('Preprocessing data...')
                X_train, X_val = preprocessor_object.remove_features(copy.deepcopy(sigma_values), z, X_train_full, X_val_full)
                for a in list(params_grid):
                    # log('Training ensemble...')
                    # logistic = LogisticRegression(solver='sag', n_jobs=NUM_THEADS, C=a['logistic__C'], tol=a['logistic__tol'])
                    # ridge = OrdinalRidge(a['ridge__alpha'])
                    # svc = LinearSVC(C=a['svc__C'], tol=a['svc__tol'])
                    # ensemble = VotingClassifier(estimators=[('logistic', logistic), ('ridge', ridge), ('svc', svc)], voting='hard', weights=[1, 1, 1])
                    # ensemble.fit(X_train, y_train)
                    # predictions_val = ensemble.predict(X_val)
                    logistic = LogisticRegression(solver='sag', n_jobs=NUM_THEADS, C=a['logistic__C'], tol=a['logistic__tol'])
                    logistic.fit(X_train, y_train)
                    predictions_val = logistic.predict(X_val)
                    mae = mean_absolute_error(predictions_val, y_val)
                    #results.append({'a_value':x, 'epsilon': y, 'reduction_level': z, 'logistic__C': a['logistic__C'], 'logistic__tol': a['logistic__tol'], 'ridge__alpha': a['ridge__alpha'], 'svc__C': a['svc__C'], 'svc__tol': a['svc__tol'], 'mae': mae})
                    results.append({'a_value':x, 'epsilon': y, 'reduction_level': z, 'logistic__C': a['logistic__C'], 'logistic__tol': a['logistic__tol'], 'mae': mae})
                    if mae < best:
                        print results[-1]
                        best = mae
    
    # Dump the results to a pickle file for later analysis
    with open('results'+sys.argv[1]+sys.argv[2], "wb") as f:
        pickle.dump(results, f)

    log('That\'s all folks!')

def main_preprop():
    """
    The main method that can be used for finding the best preprocessor hyperparameters,
    for a fixed model.
    """
    # Initialize the result array
    results = []

    # Set the search range in a parameter grid, all possible combinations will be tested
    params_preprocessor = {'a_value':range(10,15), 'epsilon':[0.1, 0.01, 0.001], 'reduction_level':[0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.5]}
    
    # Create a preprocessor object (tweaked for tuning hyperparameters)
    preprocessor_object = Preprocessor2()

    # Define the models
    logistic = LogisticRegression(solver='sag', n_jobs=NUM_THEADS, C=5, tol=0.01)
    ridge = OrdinalRidge(10)
    svc = LinearSVC(C=0.5)
    svr = svm.LinearSVR(loss='squared_epsilon_insensitive')
    lad = LAD(svr)
    ensemble = VotingClassifier(estimators=[('logistic', logistic), ('ridge', ridge), ('svc', svc), ('lad',lad)], voting='hard', weights=[1, 1, 1, 1])

    # load the data
    train_data, test_data = preprocessor_object.load()
    
    # Convert the data to tf-idf matrices
    X_train_full, X_val_full, y_train, y_val = preprocessor_object.fit_data(train_data)

    # Test the preprocessor for all possible hyperparameter combinations
    for x in params_preprocessor['a_value']:
        for y in params_preprocessor['epsilon']:
            sigma_values = preprocessor_object.compute_sigma_values(X_train_full, y_train, x, y)
            for z in params_preprocessor['reduction_level']:
                X_train, X_val, X_test = preprocessor_object.remove_features(sigma_values, z, X_train_full, X_val_full, test_data)
                ensemble.fit(X_train, y_train)
                predictions_val = ensemble.predict(X_val)
                mae = mean_absolute_error(predictions_val, y_val)
                print strftime("%H:%M:%S") +' -> a: ' + str(x) +'; epsilon: ' + str(y) + '; red lvl: ' + str(z) + '; MAE: ' + str(mae)
                results.append({'a_value':x, 'epsilon': y, 'reduction_level': z, 'mae': mae})

    print results

    log('That\'s all folks!')

def main_model():
    """
    The main method that can be used for finding the best model hyperparameters,
    for a fixed preprocessor, this method requires the data to be split in folds first.
    """

    # Set the search range in a parameter grid, all possible combinations will be tested
    params_grid = ParameterGrid({'logistic_lbfgs__C':np.logspace(-3.0, 3.0, num=5), 'logistic_lbfgs__tol':np.logspace(-3.0, 3.0, num=5),
            'logistic_lbfgs_multinom__C':np.logspace(-3.0, 3.0, num=5), 'logistic_lbfgs_multinom__tol':np.logspace(-3.0, 3.0, num=5),
            'logistic_sag_balanced__C': np.logspace(-3.0, 3.0, num=5), 'logistic_sag_balanced__tol':np.logspace(-3.0, 3.0, num=5)})

    # Create threads to run folds in parallel
    pool = ThreadPool(processes=NUM_FOLDS)
    log('starting the different folds in different threads')
    results = [pool.apply_async(single_fold_validation, (i+1, params_grid)) for i in range(NUM_FOLDS)]
    log('done')
    res = [t.get() for t in results]
    pool.close()
    pool.join()
    log('all workers are finished, pickling results')
    
    # Save the results a pickle file for further analysis
    with open('CV_results', "wb") as f:
        pickle.dump(res, f)

def single_model_tuning(modelname, fold_nr):
    """
    The thread function that can be used for finding the best model hyperparameters, for a single, non-ensemble model,
    for a fixed preprocessor, this method requires the data to be split in folds first.

    parameters:
    :param str modelname: The name of the model to test.
    :param int fold_nr: The number of the fold.
    :return list<dict> results: A list of dictionaries containing the parameter setting and the mae.
    """
    # Init a best mae so far (for printing purposes)
    best = 10
    try:
        log('Fold: '+str(fold_nr)+': Loaded the cached preprocessed data.')
        X_train, X_val, y_train, y_val, rev_val = load_fold(fold_nr)
    except IOError:
        log('Fold: '+str(fold_nr)+'run "python kfold_prepr.py" first')
    results = []

    # Tune a model based on the command line argument
    if modelname == 'log':
        par = ParameterGrid({'logistic__C': np.logspace(-5.0, 5.0, num=11), 'logistic__tol':np.logspace(-5.0, 5.0, num=11)})
        for a in list(par):
            logistic = LogisticRegression(solver='sag', n_jobs=NUM_THEADS, C=a['logistic__C'], tol=a['logistic__tol'])
            logistic.fit(X_train, y_train)
            predictions_val = logistic.predict(X_val)
            mae = mean_absolute_error(predictions_val, y_val)
            results.append({'logistic__C': a['logistic__C'], 'logistic__tol': a['logistic__tol'], 'mae': mae})
    elif modelname == 'ridge':
        par = ParameterGrid({'ridge__alpha':np.logspace(-5.0, 5.0, num=11)})
        for a in list(par):
            ridge = OrdinalRidge(a['ridge__alpha'])
            ridge.fit(X_train, y_train)
            predictions_val = ridge.predict(X_val)
            mae = mean_absolute_error(predictions_val, y_val)
            results.append({'ridge__alpha': a['ridge__alpha'], 'mae': mae})
    elif modelname == 'svc':
        par = ParameterGrid({'svc__C': np.logspace(-5.0, 5.0, num=11), 'svc__tol':np.logspace(-5.0, 5.0, num=11)})
        for a in list(par):
            svc = LinearSVC(C=a['svc__C'], tol=a['svc__tol'])
            svc.fit(X_train, y_train)
            predictions_val = svc.predict(X_val)
            mae = mean_absolute_error(predictions_val, y_val)
            results.append({'svc__C': a['svc__C'], 'svc__tol': a['svc__tol'], 'mae': mae})
    elif modelname == 'lad':
        par = ParameterGrid({'lad__C':np.logspace(-5.0, 5.0, num=11), 'lad__tol':np.logspace(-5.0, 5.0, num=11)})
        for a in list(par):
            svr_ = svm.LinearSVR(loss='squared_epsilon_insensitive')
            svr = LAD(svr_)  # use mord for rounding and clipping
            svr.fit(X_train, y_train)
            predictions_val = svr.predict(X_val)
            mae = mean_absolute_error(predictions_val, y_val)
            results.append({'lad__C': a['lad__C'], 'lad__tol': a['lad__tol'], 'mae': mae})
    elif modelname == 'final':
        # This is the tuning of the final ensemble, with fixing 0 rating predictions
        par = ParameterGrid({'logistic_lbfgs__C':np.logspace(-5.0, 5.0, num=11), 'logistic_lbfgs__tol':np.logspace(-5.0, 5.0, num=11),
            'logistic_lbfgs_multinom__C':np.logspace(-5.0, 5.0, num=11), 'logistic_lbfgs_multinom__tol':np.logspace(-5.0, 5.0, num=11),
            'logistic_sag_balanced__C': np.logspace(-5.0, 5.0, num=11), 'logistic_sag_balanced__tol':np.logspace(-5.0, 5.0, num=11)})
        
        ensemble = VotingClassifier(estimators=[
            ('logistic_lbfgs', LogisticRegression(solver='lbfgs', n_jobs=NUM_THEADS, C=5, tol=0.01)),
            ('logistic_lbfgs_multinom', LogisticRegression(solver='lbfgs', n_jobs=NUM_THEADS, C=5, tol=0.01, multi_class='multinomial')),
            ('logistic_sag_balanced', LogisticRegression(solver='sag', n_jobs=NUM_THEADS, C=5, tol=0.01, class_weight='balanced')),
            ], voting='soft', weights=[1,1,1])
        
        for a in list(par):
            ensemble.set_params(**a)
            ensemble.fit(X_train, y_train)
            predictions_val = ensemble.predict(X_val)
            predictions_val = fix_zero_predictions(predictions_val, rev_val)
            mae = mean_absolute_error(predictions_val, y_val)
            temp = a
            temp['mae'] = mae
            if mae < best:
                print temp
                best = mae
            results.append(temp)
    elif modelname == 'lbfgs_bal':
        clf = LogisticRegression(solver='lbfgs', n_jobs=NUM_THEADS, class_weight='balanced')
        par = ParameterGrid({'C':np.logspace(-1.0, 1.0, num=5), 'tol':np.logspace(-3.0, -1.0, num=3)})
        for a in list(par):
            clf.set_params(**a)
            clf.fit(X_train, y_train)
            predictions_val = clf.predict(X_val)
            predictions_val = fix_zero_predictions(predictions_val, rev_val)
            mae = mean_absolute_error(predictions_val, y_val)
            temp = a
            temp['mae'] = mae
            if mae < best:
                print temp
                best = mae
            results.append(temp)
    elif modelname == 'lbfgs_multi':
        clf = LogisticRegression(solver='lbfgs', n_jobs=NUM_THEADS, multi_class='multinomial')
        par = ParameterGrid({'C':np.logspace(-5.0, 5.0, num=11), 'tol':np.logspace(-5.0, 5.0, num=11)})
        for a in list(par):
            clf.set_params(**a)
            clf.fit(X_train, y_train)
            predictions_val = clf.predict(X_val)
            predictions_val = fix_zero_predictions(predictions_val, rev_val)
            mae = mean_absolute_error(predictions_val, y_val)
            temp = a
            temp['mae'] = mae
            if mae < best:
                print temp
                best = mae
            results.append(temp)
    elif modelname == 'sag_bal':
        clf = LogisticRegression(solver='sag', n_jobs=NUM_THEADS, class_weight='balanced')
        par = ParameterGrid({'C':np.logspace(-5.0, 5.0, num=11), 'tol':np.logspace(-5.0, 5.0, num=11)})
        for a in list(par):
            clf.set_params(**a)
            clf.fit(X_train, y_train)
            predictions_val = clf.predict(X_val)
            predictions_val = fix_zero_predictions(predictions_val, rev_val)
            mae = mean_absolute_error(predictions_val, y_val)
            temp = a
            temp['mae'] = mae
            if mae < best:
                print temp
                best = mae
            results.append(temp)
    elif modelname == 'nb':
        clf = MultinomialNB()
        par = ParameterGrid({'alpha':[0,0.01, 0.05, 0.1,0.2,0.3,0.4,0.5,1,1.5]})
        for a in list(par):
            clf.set_params(**a)
            clf.fit(X_train, y_train)
            predictions_val = clf.predict(X_val)
            predictions_val = fix_zero_predictions(predictions_val, rev_val)
            mae = mean_absolute_error(predictions_val, y_val)
            temp = a
            temp['mae'] = mae
            if mae < best:
                print temp
                best = mae
            results.append(temp)
    else:
        print "model name not defined"
        return None
    return results

def fix_zero_predictions(predictions, reviews):
    """
    Alter the predictions to introduce zero predictions based on length and ascii ascii_percentage.

    parameters:
    :param list<int> predictions: List of predicted ratings.
    :param list<str> reviews: List of the corresponding reviews' content.
    :return list<int> predictions: The altered list, with zero predictions introduced.
    """
    for x in range(len(predictions)):
        if ascii_percentage(reviews[x]) < 0.1 and len(reviews[x]) < 52 and len(reviews[x]) > 41:
            predictions[x] = 0

    return predictions

def ascii_percentage(string):
    """
    Determine the percentage of ascii characters in a string.

    parameters:
    :param str string: The string to check.
    :return float (): The fraction of ascii characters.
    """

    count = 0
    for char in string:
        if ord(char)<128:
            count += 1
    if len(string) == 0:
        return 1
    else:
        return float(count)/len(string)

def single_model_main():
    """
    The main function for finding the hyperparameters of an indiviual model.
    """
    pool = ThreadPool(processes=NUM_FOLDS)
    log('starting the different folds in different threads')

    # Start the hyperparameter gridsearch for every fold
    results = [pool.apply_async(single_model_tuning, (sys.argv[1] ,i+1)) for i in range(NUM_FOLDS)]
    log('done')
    res = [t.get() for t in results]
    pool.close()
    pool.join()
    log('all workers are finished, pickling results')
    print res
    with open('CV_results_'+str(sys.argv[1]), "wb") as f:
        pickle.dump(res, f)

def single_fold_validation(fold_nr, param_grid):
    """
    Perform a grid search of all hyperparameters for a certain fold.

    parameters:
    :param int fold_nr: The fold number.
    :param ParameterGrid param_grid: The hyperparameters to test.
    :return list<dict> results: A list of dictionaries containing the parameter setting and the mae.
    """
    try:
        log('Fold: '+str(fold_nr)+': Loaded the cached preprocessed data.')
        X_train, X_val, y_train, y_val, rev_val = load_fold(fold_nr)
    except IOError:
        log('Fold: '+str(fold_nr)+'run "python kfold_prepr.py" first')

    ensemble = VotingClassifier(estimators=[
        ('logistic_lbfgs', LogisticRegression(solver='lbfgs', n_jobs=NUM_THEADS)),
        ('logistic_lbfgs_multinom', LogisticRegression(solver='lbfgs', n_jobs=NUM_THEADS, multi_class='multinomial')),
        ('logistic_sag_balanced', LogisticRegression(solver='sag', n_jobs=NUM_THEADS, class_weight='balanced')),
        ], voting='soft', weights=[1,1,1])
    
    results = []
    best = 1
    for a in list(param_grid):
        log('Fold: '+str(fold_nr)+'Training ensemble...')
        # This is the tuning of the final ensemble, with fixing 0 rating predictions
        ensemble.set_params(**a)
        ensemble.fit(X_train, y_train)
        predictions_val = ensemble.predict(X_val)
        predictions_val = fix_zero_predictions(predictions_val, rev_val)
        mae = mean_absolute_error(predictions_val, y_val)
        temp = a
        temp['mae'] = mae
        if mae < best:
            print 'fold: ' + str(fold_nr) + ' mae: '+ str(temp)
            best = mae
        results.append(temp)
    
    return results


# Load all feature sets
def load_fold(fold_nr):
    """
    Load the preprocessed data from a certain fold.

    parameters:
    :param int fold_nr: The number of the fold.
    :return csr-matrix X_train: The feature matrix of the training samples.
    :return csr-matrix X_val: The feature matrix of the validation samples.
    :return numpy array y_train: The ratings corresponding to the training samples.
    :return numpy array y_val: The ratings corresponding to the validation samples.
    :return list<str> rev_val: A list of the validation reviews' content.
    """
    X_train = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'X_train_fold_'+str(fold_nr)+'.pkl'))
    X_val = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'X_val_fold_'+str(fold_nr)+'.pkl'))
    y_train = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'y_train_fold_'+str(fold_nr)+'.pkl'))
    y_val = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'y_val_fold_'+str(fold_nr)+'.pkl'))
    rev_val = load_pickle(os.path.join(DEFAULT_DATA_LOCATION, 'reviews_val_'+str(fold_nr)+'.pkl'))
    return X_train, X_val, y_train, y_val, rev_val


# Logging function
def log(s):
    print '[INFO] ' + s

if __name__ == '__main__':

    # Select the appropriate main method
    if MAIN_TO_RUN == 'pre':
        main_preprop()
    elif MAIN_TO_RUN == 'all':
        main_all()
    elif MAIN_TO_RUN == 'mod':
        main_model()
    elif MAIN_TO_RUN == 'single':
        single_model_main()
    else:
        print 'Not a valid value for MAIN_TO_RUN'