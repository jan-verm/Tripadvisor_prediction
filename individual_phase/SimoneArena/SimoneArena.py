
import data
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from data import load_train
from data import load_test
from matplotlib import pyplot as plt
from create_submission import write_predictions_to_csv
import math


#Loading training set
tot_train = load_train()


#Extracting the reviews' contents
tot_x_content_train = [review.content for review in tot_train][:196539]
#Creating the labels for the training set, taking them directly from the data
tot_y_train = np.array([review.rating for review in tot_train])[:196539]


import nltk
nltk.download()
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
print stop_words


# Adding very often used words in the field of hotels' reviewing to the stop-words list.
stop_words.extend([u'hotel',u'hotels',u'room',u'rooms',u'night',u'nights',u'location',u'bed',u'beds',u'place',
                   u'breakfast',u'position',u'station',u'stay',u'stayed',u'staff',u'accomodation',u'accommodations',
                  u'during',u'bathroom',u'bathrooms',u'area',u'areas',u'airport',u'reception',u'manager',u'restaurant',
                  u'floor',u'day',u'hours',u'days',u'hour',u'evening',u'morning',u'shower',u'internet',u'weekend',u'city',
                  u'service',u'spend',u'spent',u'vacation',u'holiday',u'food',u'drinks'])
# the "u" before each word; it just indicates that Python is internally representing each word as a unicode string.
print 
print stop_words


# Function to convert a raw review to a string of words
def review_to_words(raw_review):
    # 1. Removing HTML tags
    review_text = BeautifulSoup(raw_review).get_text() 
    # 2. Removing non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # 3. Converting to lower case and then splitting into individual words
    words = letters_only.lower().split()
    # 4. In Python, searching a set is much faster than searching
    #    a list, so convert the stop words to a set
    stops = set(stop_words) 
    # 5. Removing stop words
    meaningful_words = [w for w in words if not w in stops]
    # 6. Joining the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 



# Initializing an empty list to hold the clean reviews
clean_train_reviews = []
for i in range(len(tot_x_content_train)):
    clean_train_reviews.append(review_to_words(tot_x_content_train[i]))


# max_features=1200 is the maximum value that my laptop
# allowed me to insert without incurring in memory errors.
vectorizer=TfidfVectorizer(analyzer = "word", tokenizer = None,preprocessor = None, stop_words = None, max_features = 1200)


train_data_features = vectorizer.fit_transform(clean_train_reviews)
# The fit_transform method returns a sparse matrix.
# Using the toarray() function to convert this sparse representation in an ordinary one.
train_data_features = train_data_features.toarray()
#train_data_features is a matrix which has one row for each review's content
#and one column for each token occurring in the corpus of documents.


# Splitting train_data_features into 5 groups matrices each, in order to make cross validation.
a=train_data_features[:39307]
b=train_data_features[39307:78614]
c=train_data_features[78614:117921]
d=train_data_features[117921:157228]
e=train_data_features[157228:]
# Splitting tot_y_train into 5 groups of 39307 (≃196536/5) ratings each, in order to make cross validation.
a1=tot_y_train[:39307]
b1=tot_y_train[39307:78614]
c1=tot_y_train[78614:117921]
d1=tot_y_train[117921:157228]
e1=tot_y_train[157228:]


#Case 1: the 5th group is the validation set 
train_X_1 = np.vstack((a,b,c,d))
val_X_1 = e
train_Y_1 = np.concatenate((a1,b1,c1,d1))
val_Y_1 = e1


#Case 2: the 4th group is the validation set
train_X_2 = np.vstack((a,b,c,e))
val_X_2 = d
train_Y_2 = np.concatenate((a1,b1,c1,e1))
val_Y_2 = d1


#Case 3: the 3rd group is the validation set
train_X_3 = np.vstack((a,b,d,e))
val_X_3 = c
train_Y_3 = np.concatenate((a1,b1,d1,e1))
val_Y_3 = c1


#Case 4: the 2nd group is validation set 
train_X_4 = np.vstack((a,c,d,e))
val_X_4 = b
train_Y_4 = np.concatenate((a1,c1,d1,e1))
val_Y_4 = b1


#Case 5: the 1st group is the validation set
train_X_5 = np.vstack((b,c,d,e))
val_X_5 = a
train_Y_5 = np.concatenate((b1,c1,d1,e1))
val_Y_5 = a1


model = LogisticRegression()

#Case 1: Training the machine
model.fit(train_X_1,train_Y_1)
#Case 1: Predicting the labels
predictions_1 = model.predict(val_X_1)
#Case 2: Training the machine
model.fit(train_X_2,train_Y_2)
#Case 2: Predicting the labels
predictions_2 = model.predict(val_X_2)
#Case 3: Training the machine
model.fit(train_X_3,train_Y_3)
#Case 3: Predicting the labels
predictions_3 = model.predict(val_X_3)
#Case 4: Training the machine
model.fit(train_X_4,train_Y_4)
#Case 4: Predicting the labels
predictions_4 = model.predict(val_X_4)
#Case 5: Training the machine
model.fit(train_X_5,train_Y_5)
#Case 5: Predicting the labels
predictions_5 = model.predict(val_X_5)


predictions = [predictions_1, predictions_2, predictions_3, predictions_4, predictions_5]
error_counter = [0]*5
val_Y = [val_Y_1 , val_Y_2 , val_Y_3 , val_Y_4 , val_Y_5]
#Comparing the predictions with the real values
#and computing the Mean Absolute Error.
for i in range(5):
    for index in range(len(predictions[i])):
        if predictions[i][index] != val_Y[i][index]:
            error_counter[i] += math.fabs(predictions[i][index] - val_Y[i][index])
error = 0
for i in range(len(error_counter)):
    error += error_counter[i]
average_error = error / 5
print "Mean Absolute Error: " + str(float(average_error)/len(predictions[1]))


#Loading test set
test = load_test()
tot_x_content_test = [review.content for review in test]


# Initializing an empty list to hold the clean reviews
clean_test_reviews = []
for i in range(len(tot_x_content_test)):
    clean_test_reviews.append(review_to_words(tot_x_content_test[i]))



#Note that when we use the Bag of Words for the test set, we only call "transform",
#not "fit_transform" as we did for the training set
test_data_features = vectorizer.transform(clean_test_reviews)
# The fit_transform method returns a sparse matrix.
# Using the toarray() function to convert this sparse representation in an ordinary one.
test_data_features = test_data_features.toarray()
#test_data_features is a matrix which has one row for each review's content
#and one column for each token occurring in the corpus of documents.
predictions_test = model.predict(test_data_features)
#predicting the labels for the test set


write_predictions_to_csv(predictions_test, 'SimoneArena.csv')
modric = 0





