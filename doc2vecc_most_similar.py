import gensim
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import EnglishStemmer
import data
import numpy as np
from utils import dump_pickle, load_pickle
from sklearn.metrics import mean_absolute_error

LabeledSentence = gensim.models.doc2vec.LabeledSentence

TEST_SIZE = 0.2 # 0 for full model
stemmer = EnglishStemmer()
USE_BUILD_MODEL = True

class LabeledLineSentence(object):
    """
    Class that represents a document stream.
    """
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=self.stemmed_words(doc),tags=[self.labels_list[idx]])

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

def predict(model, review_content):
    """
    Predict the rating of a single review.

    parameters:
    :param Doc2Vec model: A trained doc2vec model.
    :param str review_content: The content of the review for which to predict the rating.
    :return int (): The predicted rating.
    """
    dv = model.infer_vector(list(stemmer.stem(w) for w in gensim.utils.tokenize(review_content, lower=True, deacc = True)))
    return int(model.docvecs.most_similar(positive=[dv])[0][0])

def predict_val_set(model, docs_val):
    """
    Perform the predict function for all validation reviews.

    parameters:
    :param Doc2Vec model: A trained doc2vec model.
    :param list<str> docs_val: A list of the validation reviews' content.
    :return list<int> predictions: A list of the predicted ratings.
    """
    count = 0 
    predictions = []
    for doc in docs_val:
        predictions.append(predict(model, doc))
        count +=1
        if count % 1000 == 0 :
            log('predicted '+str(count)+' values!')

    return predictions

def main():
    # Load the dataset
    data_set = data.load_pickled_data()
    train_data = data_set['train']
    test_data = data_set['test']
    log('loaded dataset!')
    traindocs = [doc.content for doc in train_data if int(doc.rating) != 0]
    trainlabels = [int(doc.rating) for doc in train_data if int(doc.rating) != 0]
    
    # Split the dataset
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
        d2v = gensim.models.Doc2Vec.load('Models/doc2vec_val.model')

    # Predict
    val_predictions = predict_val_set(d2v, docs_val)

    # Print the mae
    print 'MAE on validation set: ' + str(mean_absolute_error(label_val,val_predictions))

# Logging function
def log(s):
    print '[INFO] ' + s

if __name__ == '__main__':
    main()