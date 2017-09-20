import data
import numpy as np
import math
import scipy
import csv
import heapq
import os.path
import sys
from mord import OrdinalRidge
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from time import strftime
from scipy.sparse import csr_matrix
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

# Initialize global parameters
reviews_train = {}
reviews_test = {}
pred_dict = {}

# Helperclass, used to store feature indexes, and their sigma-values as priority
class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]


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
    C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
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


# Compute the difference between two lists
def list_diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]


def read_train_csv_files():
    """
    Read the preprocessed training data.
    imput arguments:
        None
    output arguments:
        None
    """
    for language in language_list:
        if os.path.isfile('Data/preprocessed/reviews_'+language+'.csv'): 
            reviews_train[language]=[]
            with open('Data/preprocessed/reviews_'+language+'.csv',mode='r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    read_rev = data.Review(id=row['id'], author=row['author'], content=row['content'], date=row['date'], rating=row['rating'],subratings=row['subratings'], hotel=row['hotel'])
                    reviews_train[language].append(read_rev)
        else:
            print "missing a preprocessed file: " +language + "."


def read_test_csv_files():
    """
    Read the preprocessed test data.
    imput arguments:
        None
    output arguments:
        None
    """
    for language in language_list:
        if os.path.isfile('Data/preprocessed/reviews_test_'+language+'.csv'): 
            reviews_test[language]=[]
            with open('Data/preprocessed/reviews_test_'+language+'.csv',mode='r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    read_rev = data.Review(id=row['id'], author=row['author'], content=row['content'], date=row['date'], rating=row['rating'],subratings=row['subratings'], hotel=row['hotel'])
                    reviews_test[language].append(read_rev)        
        else:
            print "missing a preprocessed file: " +language + "."


def feature_extraction(review_list):
    """
    Apply feature extraction from the reviews.
    imput arguments:
        review_list: a list of Review items
    output arguments:
        TFIDF_normed: a sparse csr matrix containing the normalized TF-IDF features extracted from the reviews
        rating_list: a list ratings corresponding to the reviews
        tfidf_transformer: a TfidfTransformer object, for later usage on the test reviews (contains normalization parameters)
        count_vectorizer.vocabulary: a dictionary containing the extracted words and their feature index
    """

    # Get all content and all ratings
    content_list = [o.content for o in review_list]
    rating_list = [int(o.rating) for o in review_list]

    # Get TF's (term frequencies)
    count_vectorizer = CountVectorizer(decode_error='replace', min_df = 2)
    X_content = count_vectorizer.fit_transform(content_list)

    # Balance the data (this is omitted deu to complexity constraints)
    # X_content, rating_list = balance_data(X_content, ratings)

    # Transform the TF's to TFIDF's (Term Frequency Inverse Document Frequency)
    tfidf_transformer = TfidfTransformer(smooth_idf=False)
    tfidf_transformer.fit(X_content)
    TFIDF_normed = tfidf_transformer.transform(X_content)

    # return TFIDF_normed, rating_list, IDF_for_normalization, count_vectorizer.vocabulary_
    return TFIDF_normed, rating_list, tfidf_transformer, count_vectorizer.vocabulary_


def reduce_feature_space(normed_TFIDF_features, a_value, rating_list, fitted_transformer, vocabulary):
    """
    Reduce the number of features that will be used to train the model.
    imput arguments:
        normed_TFIDF_features: a sparse csr matrix containing the normalized TF-IDF features extracted from the reviews
        a_value: a hyperparameter to tune, see report for more information
        rating_list: a list ratings corresponding to the reviews ( rows of the feature matrix)
        fitted_transformer: a TfidfTransformer object, to extract IDF values from
        vocabulary: a dictionary containing the used words and their feature index
    output arguments:
        reduced_normed_TFIDF_features: a sparse csr matrix containing the reduction of thenormalized TF-IDF features
        reduced_vocabulary: a dictionary containing the reduced set of words and their feature index
        new_fitted_transformer: a TfidfTransformer object, from which some IDF-values have been removed
    """

    # Initialize the sigma priority queues
    sigma_values = {0:PriorityQueue(),1:PriorityQueue(),2:PriorityQueue(),3:PriorityQueue(),4:PriorityQueue(),5:PriorityQueue()}

    # Set the reduction level, this is a hyperparameter, but is not tuned in the first phase, a value of 10% is taken from literature
    reduction_level = 0.10

    # The number of features is the number of columns in the feature matrix
    all_features = normed_TFIDF_features.shape[1]

    # Keep "reduction_level*100"% of all features
    keep_features = int(math.ceil(float(all_features)*reduction_level))

    # Number of features that may be chosen per rating
    features_per_rating = int(math.ceil(float(keep_features)/6))

    # Get IDF values from the transformer
    IDFs = fitted_transformer._idf_diag.diagonal()

    # Loop over all features

    # Use a counter to visualize the progress
    count = 1.0
    prev = 0.0
    epsilon = 0.01

    # Transform matrix to csc to perform more effectively with column operations
    normed_TFIDF_features = normed_TFIDF_features.tocsc()

    # Loop over all features
    for i in range(0, all_features):
        feature = np.ndarray.flatten(normed_TFIDF_features.getcol(i).toarray())
        rating_list = numpy.array(rating_list)
        
        # Compute sigma value as described in literature
        sigma_set = rating_list[feature==0]
        mean_rating = int(math.round(sigma_set.mean()))
        
        # Do not invert this value, as is done in literature, this makes using heapq easier
        sigma_value = (sigma_set.var()+epsilon)*((IDFs[i])**a_value)
        
        # Store feature index at the rating with nearest mean
        sigma_values[mean_rating].put(i, sigma_value)
        
        # Print "counter" for visualizing progress
        if count/all_features > prev:
            print strftime("%H:%M:%S") +' : ' + str(math.ceil(100*(count/all_features*100))/100) + '% of the features checked for sigma values.'
            prev += 0.01
        count +=1.0

    # Selection of the most informative features
    features_to_keep = []
    
    # Transform back into csr format (to delete feature cols)
    normed_TFIDF_features = normed_TFIDF_features.tocsr()

    # Let every rating class choose a feature in a round robin fashion
    for a in range(0, 6*features_per_rating):
        if not sigma_values[a%6].empty():
            features_to_keep.append(sigma_values[a%6].get())

    # Remove all features that will not be used
    remove = list_diff(range(0, all_features), features_to_keep)
    reduced_normed_TFIDF_features = drop_cols( normed_TFIDF_features,remove)

    # Remove all words from the vocabulary that won't be used, adept indices
    reduced_vocabulary_temp = {k:v for (k,v) in vocabulary.items() if v in features_to_keep}
    sorted_voc = sorted(reduced_vocabulary_temp.items(), key=lambda x: x[1])
    reduced_vocabulary = {}
    for z in range(0, len(reduced_vocabulary_temp)):
        reduced_vocabulary[sorted_voc[z][0]]=z

    new_fitted_transformer = TfidfTransformer(smooth_idf=False)
    new_fitted_transformer._idf_diag = fitted_transformer._idf_diag.copy()
    
    # Remove unnecessary IDF values from the IDF matrix of the fitted transform
    new_fitted_transformer._idf_diag = drop_cols(new_fitted_transformer._idf_diag, remove)
    new_fitted_transformer._idf_diag = drop_rows(new_fitted_transformer._idf_diag, remove)

    return reduced_normed_TFIDF_features, reduced_vocabulary, new_fitted_transformer


def determine_hyper_parameter(review_list, k):
    """
    Use k-fold CV to determine the values of a (to reduce dimensionality) and alpha (ridge parameter).
    imput arguments:
        review_list: a list of Review items
        k: determines in how many parts the training data should be split
    output arguments:
        mean_a: the best a-value (averaged out over k)
        mean_alpha: the best alpha-value (averaged out over k)
        results: list of list of 3-folded tuples, each inner list represents an iteration in the k-fold CV,
                 each tuple within the innner list is structured as: (a-value, alpha-value, mae)
    """

    # Initialize the a-values and alpha-values to test
    a_range = range(1,11)
    alpha_range = [0.01, 0.1, 1, 10, 100, 1000, 10000]

    # Split training set in k parts
    kf = KFold(n_splits=k)

    # Keep the results
    results = []

    # Loop over all k-folds
    for train, test in kf.split(review_list):

        # Seperate train list from test list
        train_list = [review_list[i] for i in train]
        test_list = [review_list[i] for i in test]
        actual_ratings = [int(r.rating) for r in test_list]

        # Extract the features from the training set
        TFIDF_normed, rating_list, fit_transformmer, vocabulary = feature_extraction(train_list)

        # The inner list (1 per iteration in the k-fold CV)
        item = []
        for a_value in a_range:

            # Reduce the featurespace, based on the value of a
            reduced_normed_TFIDF_features, reduced_vocabulary, fitted_transformer = reduce_feature_space(TFIDF_normed, a_value, rating_list, fit_transformmer, vocabulary)

            for alpha_value in alpha_range:
                
                # Train the model for given features and given alpha
                OR = train_model_feature_input(reduced_normed_TFIDF_features, rating_list, alpha_value)
                
                # Use model to predict the ratings from the testset, and compute MAE
                my_predictions = []
                for rev in test_list:
                    my_predictions.append(int(predict_rating(rev, OR, reduced_vocabulary, fitted_transformer)))

                MAE = mean_absolute_error(actual_ratings, my_predictions)
                # Print a_value, alpha_value, MAE for inspection during k-fold CV
                # print a_value, alpha_value, MAE
                item.append((a_value,alpha_value, MAE))
        results.append(item)

    # Look for the smallest mae in each iteration of the k-fold CV
    best = []
    for x in results:
        best.append((min(x, key = lambda t: t[2])[0],min(x, key = lambda t: t[2])[1]))

    # Compute the averages of the hyperparametes that were used for the smallest mae-values
    mean_a = np.mean([x[0] for x in best])
    mean_alpha = np.mean([x[1] for x in best])

    return mean_a, mean_alpha, results


def train_model(review_list, a_value, alpha_value=1.0):
    """
    Train an ordinal ridge regression model, with a given a-value and given alpha-value.
    imput arguments:
        review_list: a list of Review items
        a_value: a hyperparameter to tune, see report for more information
        alpha_value: a hyperparameter to use in ridge regression (for penalizing big weigths)
    output arguments:
        OR: the trained model
        fitted_transformer: a TfidfTransformer object, for later usage on the test reviews (contains normalization parameters)
        reduced_vocabulary: a dictionary containing the reduced set of words and their feature index
    """

    print "start training"
    # Extract the features
    TFIDF_normed, rating_list, IDFs, vocabulary = feature_extraction(review_list)
    print "features extracted"
    # Reduce the dimensionality
    reduced_normed_TFIDF_features, reduced_vocabulary, fitted_transformer = reduce_feature_space(TFIDF_normed, a_value, rating_list, IDFs, vocabulary)
    print "featurespace reduced"

    # Initialize the model
    OR = OrdinalRidge(alpha=alpha_value)
    # Train the model
    OR.fit(reduced_normed_TFIDF_features, rating_list)

    return OR, fitted_transformer, reduced_vocabulary


def train_model_feature_input(features, ratings, alpha_value):
    """
    Train an ordinal ridge regression model, with a given alpha-value.
    imput arguments:
        featues: normalized, reduced tf-idf feature matrix
        ratings: a list ratings corresponding to the reviews (rows of the feature matrix)
        alpha_value: a hyperparameter to use in ridge regression (for penalizing big weigths)
    output arguments:
        OR: the trained model
    """

    # Initialize the model
    OR = OrdinalRidge(alpha=alpha_value)
    # Train the model
    OR.fit(features, ratings)
    print "model trained on features"
    return OR


def predict_rating(review, model, reduced_vocabulary, fitted_transformer):
    """
    Predict the rating for a single review.
    imput arguments:
        review: Review item for which the rating must be predicted
        model: a ML model used for prediction
        reduced_vocabulary: a dictionary containing the reduced set of words and their feature index
        fitted_transformer: a TfidfTransformer object, for usage on the test reviews (contains normalization parameters)
    output arguments:
        prediction: the predicted rating
    """
    # Determine the TFDIF representation of the review to predict
    count_vectorizer = CountVectorizer(decode_error='replace', vocabulary=reduced_vocabulary, ngram_range = (1,2))
    X_content = count_vectorizer.fit_transform([review.content])
    
    # Use the fitted_transformer to use the same normalization as on the training data
    try:
        normed_X_content_IDF = fitted_transformer.transform(X_content)
        return model.predict(normed_X_content_IDF)
    # If content is None, predict 0
    except:
        return 0


def make_csv_prediction_file(prediction_dict):
    """
    Save the predictions to a csv-file.
    input arguments:
        prediction_dict: a dictionary with languages as keys, and a list of tuples (id, prediction) as value
    output arguments:
        None
    """

    # Merge all predictions together and sort them based on id
    all_predictions = []
    for k,v in prediction_dict.iteritems():
        all_predictions.append(v)
    flat_predictions = [item for sublist in all_predictions for item in sublist]
    non_array_pred = [(int(rev[0]),int(rev[1])) for rev in flat_predictions]
    predictions = sorted(flat_predictions, key=lambda pred: int(pred[0]))

    with open('predictions.csv','wb') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['id','rating'])
        for rev_id, rev_rating in predictions:
            csv_out.writerow([rev_id,rev_rating[0]])


def predict_list_of_reviews(language, reviews, model, reduced_vocabulary, fitted_transformer):
    """
    Predict the rating for a lis of reviews.
    imput arguments:
        language: the language of the reviews in the list
        reviews: a list of Review items for which the rating must be predicted
        model: a ML model used for prediction
        reduced_vocabulary: a dictionary containing the reduced set of words and their feature index
        fitted_transformer: a TfidfTransformer object, for usage on the test reviews (contains normalization parameters)
    output arguments:
        None, the ouput is stored in a global variable
    """

    pred_list = []
    # Predict the rating for each review in the list
    for rev in reviews:
        rat = predict_rating(rev, model, reduced_vocabulary, fitted_transformer)
        pred_list.append((rev.id, rat))
    pred_dict[language] = pred_list


def balance_data(X_content, ratings):
    """
    Balance the training data, first apply oversampling (SMOTE) afterwards clean the data/undersample (ENN)
    imput arguments:
        X_content: The full feature matrix, not yet transformed to TFIDF format
        ratings: The corresponding ratings
    output arguments:
        return_csr: The balanced X_content
        return_ratings: The balanced, corresponding ratings
    """

    # Initialize SMOTE object for oversampling and ENN object for cleaning the oversampled data
    sm = SMOTE()
    enn = EditedNearestNeighbours()
    nr_revs = X_content.shape[0]
    
    # Handle content in 20 parts to avoind Memory errors!
    return_csr = csr_matrix((0, X_content.shape[1]))
    return_ratings = []
    nr_chuncks = 20
    chunck = nr_revs/nr_chuncks
    for x in range(0,nr_chuncks):
        # Get appropriot part of the data
        if x < nr_chuncks-1:
            X_now = X_content[x*chunck:(x+1)*chunck, :].toarray()
            ratings_now = ratings[x*chunck:(x+1)*chunck]
        else:
            X_now = X_content[x*chunck:nr_revs, :].toarray()
            ratings_now = ratings[x*chunck:nr_revs]

        # Apply SMOTE for each minority class
        for i in range(0,4):
            X_now, ratings_now = sm.fit_sample(X_now, ratings_now)

        # Apply ENN for cleaning
        X_now, ratings_now = enn.fit_sample(X_now, ratings_now)

        # Append data to the return matrix
        vstack([return_csr,csr_matrix(X_now)])
        return_ratings.extend(ratings_now)

    print "balanced"
    return return_csr, return_ratings


if __name__ == "__main__":

    if sys.argv[1] == 'lang':
        language_list = ['en','de','fr','it','es','zh']
    elif sys.argv[1] == 'all':
        language_list = ['all']
    else:
        raise('not a valid option, usage: python review_classification_prediction.py option; option = \"lang\" or \"all\"')
    read_train_csv_files()
    print "train reviews loaded!"
    read_test_csv_files()
    print "test reviews loaded!"

    # After CV, these hyperparameters turned out to be the best (reduction level was not tuned, fixed a 10%)
    a_values = {'en': 6,'de':8,'fr': 10,'it':5,'es':6,'zh':9, 'all':6}
    alpha_values = {'en': 10,'de':10,'fr':1,'it':1,'es':1,'zh':0.1, 'all':10}

    for k,v in reviews_train.iteritems():
        model, fitted_transformer, reduced_vocabulary = train_model(v, a_values[k], alpha_values[k])
        predict_list_of_reviews(k, reviews_test[k], model, reduced_vocabulary, fitted_transformer)

    make_csv_prediction_file(pred_dict)

