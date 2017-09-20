import gensim
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import EnglishStemmer
from sklearn.linear_model import LogisticRegression
import data
import numpy as np
from utils import dump_pickle, load_pickle
from sklearn.metrics import mean_absolute_error, classification_report
from scipy.sparse import csr_matrix, vstack
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

LabeledSentence = gensim.models.doc2vec.LabeledSentence

TEST_SIZE = 0.2 # 0 for full model
stemmer = EnglishStemmer()
USE_BUILD_MODEL = False
BALANCE = False

class LabeledLineSentence(object):
    """
    Class that represents a document stream.
    """
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=self.stemmed_words(doc),tags=[idx])

    def stemmed_words(self, doc):
        return list(stemmer.stem(w) for w in gensim.utils.tokenize(doc, lower=True, deacc = True))

def train_model(iterator):
    """
    Train the doc2vec model.

    parameters:
    :param LabeledLineSentence__iterator iterator: An interator representing a document stream.
    :return Doc2Vec model: A trained doc2vec model.
    """
    model = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=8,alpha=0.025, min_alpha=0.025)
    log('building vocabulary')
    model.build_vocab(iterator)
    for epoch in range(10):
    	log('actual traing of the network, pass: ' + str(epoch))
        model.train(iterator)
        model.alpha -= 0.002 # decrease the learning rate
        model.min_alpha = model.alpha # fix the learning rate, no deca
        model.save('Models/doc2vec'+str(TEST_SIZE)+'.model')
    return model

def fittransform_feature_matrix(model):
    """
    Transform the doc2vec feature matrix to a csr_matrix.

    parameters:
    :param Doc2Vec model: A trained doc2vec model.
    :return csr-matrix (): The csr feature matrix.
    """
    return csr_matrix(np.matrix(model.docvecs))


def transform_feature_matrix(model, test_reviews):
    """
    Transform the test/validation reviews to vectors and store them in a csr_matrix.

    parameters:
    :param Doc2Vec model: A trained doc2vec model.
    :param list<str> test_reviews: A list of test/validation reviews' content.
    :return csr-matrix matrix: The test/validation feature matrix.
    """
    matrix = csr_matrix(model.infer_vector(list(stemmer.stem(w) for w in gensim.utils.tokenize(test_reviews[0], lower=True, deacc = True))))
    for i in range(1, len(test_reviews)):
        if count/float(all_rev) > done:
            log('transformed '+str(count)+' of '+str(all_rev)+' test reviews!')
            done += 0.1
        matrix = vstack([matrix, csr_matrix(model.infer_vector(list(stemmer.stem(w) for w in gensim.utils.tokenize(test_reviews[i], lower=True, deacc = True))))])
        count += 1
    return matrix


def main():
    # Load the data
    data_set = data.load_pickled_data()
    train_data = data_set['train']
    test_data = data_set['test']
    log('loaded dataset!')
    traindocs = [doc.content for doc in train_data if int(doc.rating) != 0]
    trainlabels = [int(doc.rating) for doc in train_data if int(doc.rating) != 0]
   
    # Split the data
    if TEST_SIZE > 0:
        log('split dataset...')
        docs_train, docs_val, label_train, label_val = train_test_split(traindocs, trainlabels, test_size=TEST_SIZE, random_state=0)
    else:
        docs_train = traindocs
        label_train = trainlabels
    
    # Use prebuild model
    if not USE_BUILD_MODEL:
        log('make iterator...')
        it = LabeledLineSentence(docs_train, label_train)
        log('start training NN')
        d2v = train_model(it)
    else:
        log('load pretrained model')
        d2v = gensim.models.Doc2Vec.load('Models/doc2vec0.2.model')

    train_features = fittransform_feature_matrix(d2v)
    log('start computing vectors for test data...')
    val_features = transform_feature_matrix(d2v, docs_val)

    # Actual classification
    logistic = LogisticRegression(solver='sag', n_jobs=4, C=1, tol=0.1)
    logistic.fit(train_features, label_train)
    predictions = logistic.predict(val_features)
    log('Validation error = %s' % str(mean_absolute_error(predictions, label_val)))
    log(classification_report(predictions, label_val))

# Logging function
def log(s):
    print '[INFO] ' + s

if __name__ == '__main__':
    main()
