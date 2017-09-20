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

'''

from preprocessor import *
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import *
from sklearn.metrics import mean_absolute_error
from postprocessor import fix_zero_predictions
import matplotlib
matplotlib.use('pdf')
import itertools

from sklearn.svm.classes import LinearSVC

NUM_THREADS = 4
BEST_A = 11
BEST_EPSILON = 0.1
MAX_NUMBER_OF_MODELS = 1
USE_CACHED_FEATURES = True


def execute_ensemble(ensemble, information):
    """
    Execute the given ensemble and print out its score. Notify if this is a new
    best score.

    :param ensemble: The ensemble to execute
    :param information: The information to print together with the validation score.
    :return: The validation score
    """
    ensemble = ensemble.fit(X_train, y_train)
    predictions_val = ensemble.predict(X_val)
    predictions_val = fix_zero_predictions(predictions_val, preproc.val_reviews)
    score = mean_absolute_error(predictions_val, y_val)

    new_best_score_string = ""
    global BEST_SCORE
    if score < BEST_SCORE:
        BEST_SCORE = score
        new_best_score_string = " NEW BEST SCORE"

    log("{} - {}".format(score, information + new_best_score_string))
    return score


def ensemble_combinations():
    """
    Create a list of models to combine them in an ensemble. Generate all possible
    combinations with a given max size and test them. The top results are printed
    at the end of the execution.
    """
    component_array = [
        ('logistic_lbfgs',
            LogisticRegression(solver='lbfgs', n_jobs=NUM_THREADS, C=5, tol=0.01)),
        ('logistic_lbfgs_multinom',
            LogisticRegression(solver='lbfgs', n_jobs=NUM_THREADS, C=5, tol=0.01, multi_class='multinomial')),
        ('logistic_lbfgs_multinom_balanced',
            LogisticRegression(solver='lbfgs', n_jobs=NUM_THREADS, C=5, tol=0.01, multi_class='multinomial', class_weight='balanced')),
        ('logistic_sag',
            LogisticRegression(solver='sag', n_jobs=NUM_THREADS, C=5, tol=0.01)),
        ('logistic_sag_multinom',
            LogisticRegression(solver='sag', n_jobs=NUM_THREADS, C=5, tol=0.01, multi_class='multinomial')),
        ('lin_svc', LinearSVC(C=0.5)),
        ('multinomial', MultinomialNB(alpha=0.15)),
        ('logistic_sag_balanced',
            LogisticRegression(solver='sag', n_jobs=NUM_THREADS, C=5, tol=0.01, class_weight='balanced')),
        ('logistic_lbfgs_balanced',
            LogisticRegression(solver='lbfgs', n_jobs=NUM_THREADS, C=5, tol=0.01, class_weight='balanced')),
    ]

    results = []
    for i in range(1, MAX_NUMBER_OF_MODELS + 1):  # amount of models in the ensemble
        subsets = itertools.combinations(component_array, i) # get all possible combinations of this length
        for subset in subsets:
            names = ",".join([x[0] for x in subset])

            try:
                # soft voting
                try:
                    ensemble = VotingClassifier(estimators=subset, voting='soft')
                    score = execute_ensemble(ensemble, names + ' (soft)')
                    results.append((score, names))
                except AttributeError:  # ensemble isn't able to use soft voting, just execute the next ensemble
                    pass

                # hard voting
                ensemble = VotingClassifier(estimators=subset, voting='hard')
                score = execute_ensemble(ensemble, names + ' (hard)')
                results.append((score, names))

            except MemoryError:  # catch memory errors, just execute the next ensemble
                log("out of memory - {}".format(names))

    # print out the best ensembles
    print("\n ---BEST RESULTS---")
    best_results = sorted(results)[0:10]
    for result in best_results:
        print("{} - {}".format(result[0], result[1]))

    log('That\'s all folks!')


if __name__ == '__main__':

    log('Preprocessing data...')
    preproc = Preprocessor(a_value=BEST_A, epsilon=BEST_EPSILON, use_cached_features=USE_CACHED_FEATURES)
    X_train, X_val, y_train, y_val, X_test = preproc.load_and_preprocess()

    BEST_SCORE = 5
    ensemble_combinations()
