from preprocessor import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from imblearn.over_sampling import SMOTE
from scipy.sparse import csr_matrix, vstack
import numpy as np
from operator import itemgetter

NUM_ITER = 16
BAL = False
NORM = True

def main():
    """
    Load and preprocess the data, afterwards it can both scale and or balance the input feature matrix.
    Use classifier and print mae.
    """
    log('Preprocessing data...')
    preproc = Preprocessor(a_value=11, epsilon=0.1, use_cached_features=True, reduction_level=0.025)
    X_train, X_val, y_train, y_val, X_test = preproc.load_and_preprocess()

    if NORM:
        X_train, X_val = scale_features(X_train, X_val)
    if BAL:
        X_train, y_train = balance_data(X_train, y_train)

    clf = LogisticRegression(solver='sag', n_jobs=4, C=5, tol=0.01)
    clf.fit(X_train, y_train)

    log('MAE: ' + str(mean_absolute_error(clf.predict(X_val), y_val)))

    log('That\'s all folks!')

def balance_data(X_train, y_train):
    """
    Balance the input data, so that all classes occur in a simular quantity.
    
    parameters:
    :param csr-matrix X_train: The preprocessed tf-idf feature matrix of the training samples. 
    :param numpy array y_train : The ratings corresponding to the samples in the training feature matrix.
    :return csr-matrix X_train_bal: The balanced feature matrix.
    :return numpy array y_train_bal: The ratings corresponding to the samples in the balanced validation feature matrix.
    """
    num_rev = X_train.shape[0]
    X_train_bal = csr_matrix((0,X_train.shape[1]))
    y_train_bal = []
    chunck = num_rev/NUM_ITER

    #Split data in chuncks, to fit them in to memory
    for a in range(0,NUM_ITER):
        if a == NUM_ITER-1:
            end = num_rev
        else:
            end = (a+1)*chunck
        
        X_dense = X_train[a*chunck:end,:].todense()

        log('Balancing...')
        sm = SMOTE(kind='regular')
        X_resampled, y_resampled = sm.fit_sample(X_dense, y_train[a*chunck:end])
        for epoch in range(2):
            X_resampled, y_resampled = sm.fit_sample(X_resampled, y_resampled)

        X_train_bal = vstack([X_train_bal, csr_matrix(X_resampled)])
        y_train_bal.extend(y_resampled)

    return X_train_bal, y_train_bal

def scale_features(X_train, X_val):
    """
    Determine scaling parameters for the training features.
    Transform the traing features and the validation features using the scaling parameters.

    parameters:
    :param csr-matrix X_train: The preprocessed tf-idf feature matrix of the training samples.
    :param csr-matrix X_val: The preprocessed tf-idf feature matrix of the validation samples.
    :return csr-matrix normed_X_train: The scaled feature matrix of the training samples.
    :return csr-matrix normed_X_val: The scaled feature matrix of the validation samples.
    """
    scaler = StandardScaler(copy=True, with_mean=False, with_std=True)
    normed_X_train = scaler.fit_transform(X_train)
    normed_X_val = scaler.transform(X_val)

    return normed_X_train, normed_X_val

# Logging function
def log(s):
    print '[INFO] ' + str(s)


if __name__ == '__main__':
    main()