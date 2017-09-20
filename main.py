"""
Created on 14-nov.-2016

main.py is used to assess the results of the optimal model found in 
ensemble.py and using the parameters found by preprocessor_tuning.py
and tune_hyperpar.py.
Different error metrics for this model and its result are printed, 
to be able to decide on further improvements.

Required folder structure:
 |-- Data
 |   |-- Test
 |   |-- Train
 |-- Predictions
 |-- utils.py
 |-- data.py
 |-- preprocessor.py

"""

from preprocessor import *
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import itertools
import utils
import data

from postprocessor import fix_zero_predictions

NUM_THREADS = 4
BEST_A = 11
BEST_EPSILON = 0.1
USE_CACHED_FEATURES = False


def main():
    """
    This main flow of the program: 
    - Data is read and preprocessed
    - An ensemble model is trained using the train data
    - The validation set is predicted, different error metrics are printed
    - The test set is predicted and predictions are written to a file
    """
    log('Preprocessing data...')
    preproc = Preprocessor(a_value=BEST_A, epsilon=BEST_EPSILON, use_cached_features=USE_CACHED_FEATURES)
    X_train, X_val, y_train, y_val, X_test = preproc.load_and_preprocess()

    log('Training ensemble...')
    ensemble = VotingClassifier(estimators=[
        ('multinomial', MultinomialNB(alpha=0.01)),
        ('logistic_sag_balanced',
         LogisticRegression(solver='sag', n_jobs=NUM_THREADS, C=5, tol=0.01, class_weight='balanced')),
        ('logistic_lbfgs_balanced',
         LogisticRegression(solver='lbfgs', n_jobs=NUM_THREADS, C=5, tol=0.01, class_weight='balanced')),
    ], voting='soft', weights=[1, 1, 1])
    ensemble = ensemble.fit(X_train, y_train)

    # Uncomment when using a test_size > 0 in preprocessor.py
    # log('Predicting validation set...')
    # predictions_val = ensemble.predict(X_val)
    # if USE_CACHED_FEATURES:
    #     reviews = preproc.val_reviews
    # else:
    #     reviews = preproc.load_val_reviews()
    # predictions_val = fix_zero_predictions(predictions_val, reviews)
    # log('Validation error = %s' % str(mean_absolute_error(predictions_val, y_val)))
    # log(classification_report(predictions_val, y_val))
    # plot_confusion_matrix(confusion_matrix(y_val, predictions_val), classes=[1, 2, 3, 4, 5],
    #                       title='Normalized confusion matrix: validation set', filename='Plots/val_cnf_matrix.pdf')

    log('Predicting test set...')
    test_reviews = data.load_pickled_data()['test']
    test_content = [x.content for x in test_reviews]
    predictions_test = ensemble.predict(X_test)
    predictions_test = fix_zero_predictions(predictions_test, test_content)

    pred_file_name = utils.generate_unqiue_file_name(PREDICTIONS_BASENAME, 'csv')
    log('Dumping predictions to %s...' % pred_file_name)
    write_predictions_to_csv(predictions_test, pred_file_name)

    log('That\'s all folks!')



def log(s):
    """ Logging function """
    print '[INFO] ' + str(s)


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


def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues,filename='Plots/plot.pdf'):
    """
    This function prints and plots the normalized confusion matrix..
    It will be saved to a pdf-file in the working directory.
    """
    fig = plt.figure()

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{0:.2e}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fig.savefig(filename)

if __name__ == '__main__':
    main()
