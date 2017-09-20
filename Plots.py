import data
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from data import load_train
import langid
from collections import Counter

train_data = load_train()

#first item in list is enchant dictionary key, second item is nltk stemming language
language_map = {'en':['en','english'],'de':['de_DE','german'],'fr':['fr','french'],'it':['it_IT','italian'],'es':['es','spanish']}
language_list = ['en','de','fr','it','es','zh']

def split_by_language(reviews):
    """
    Split the reviews based on their language.
    input arguments:
        reviews: a list of review items
    output arguments:
        reviews_dict_languages: a dictionary with languages as keys, 
                                and a list of the corresponding reviews as value.
    """

    # Initialization
    reviews_dict_languages = {}
    langid.set_languages(language_list)

    # Use a counter to visualize the progress
    count = 1

    # Loop over all reviews
    for review in reviews:

        # Detect the language
        language = langid.classify(review.content)[0]

        #Store the review in the corresponding dictionary by language
        if language in reviews_dict_languages:
            reviews_dict_languages[language].append(review)
        else:
            reviews_dict_languages[language] = []
            reviews_dict_languages[language].append(review)

    return reviews_dict_languages


content_lenghts_1 = []   # content lenghts of the reviews whose rating is 1
content_lenghts_2 = []   # content lenghts of the reviews whose rating is 2
content_lenghts_3 = []
content_lenghts_4 = []
content_lenghts_5 = []

non_ASCII_characters_1 = 0   # number of rated 1 reviews which contain non-ASCII characters
non_ASCII_characters_2 = 0
non_ASCII_characters_3 = 0
non_ASCII_characters_4 = 0
non_ASCII_characters_5 = 0

reviews_1 = [] # reviews whose rating is 1
reviews_2 = [] # reviews whose rating is 2
reviews_3 = []
reviews_4 = []
reviews_5 = []

prices_1 = []  # hotel prices for reviews whose rating is 1
prices_2 = []  # hotel prices for reviews whose rating is 2
prices_3 = []
prices_4 = []
prices_5 = []

A_TripAdvisor_member_counter_1 = 0   # number of rated 1 reviews by 'A TripAdvisor member'
A_TripAdvisor_member_counter_2 = 0   # number of rated 2 reviews by 'A TripAdvisor member'
A_TripAdvisor_member_counter_3 = 0   # number of rated 3 reviews by 'A TripAdvisor member'
A_TripAdvisor_member_counter_4 = 0
A_TripAdvisor_member_counter_5 = 0
Lass_counter_1 = 0      # number of rated 1 reviews by 'lass='
Lass_counter_2 = 0      # number of rated 2 reviews by 'lass='
Lass_counter_3 = 0
Lass_counter_4 = 0
Lass_counter_5 = 0

for review in train_data:
    if review.rating==1:
        reviews_1.append(review)
        content_lenghts_1.append(len(review.content))
        if review.subratings.location != -1:
            locations_1.append(review.subratings.location)
        if review.subratings.cleanliness != -1:
            cleanliness_1.append(review.subratings.cleanliness)
        if review.hotel.price != -1 :
            prices_1.append(review.hotel.price)
        if review.author == 'A TripAdvisor Member' :
            A_TripAdvisor_member_counter_1 += 1
        if review.author == 'lass=' :
            Lass_counter_1 += 1
        if not(all(ord(char) < 128 for char in review.content)):
            non_ASCII_characters_1 += 1
          
            
    elif review.rating==2:
        reviews_2.append(review)
        content_lenghts_2.append(len(review.content))
        if(review.subratings.location != -1):
            locations_2.append(review.subratings.location)
        if(review.subratings.cleanliness != -1):
            cleanliness_2.append(review.subratings.cleanliness)
        if review.hotel.price != -1 :
            prices_2.append(review.hotel.price)
        if review.author == 'A TripAdvisor Member' :
            A_TripAdvisor_member_counter_2 += 1
        if review.author == 'lass=' :
            Lass_counter_2 += 1
        if not(all(ord(char) < 128 for char in review.content)):
             non_ASCII_characters_2 += 1
        
    elif review.rating==3:
        reviews_3.append(review)
        content_lenghts_3.append(len(review.content))
        if(review.subratings.location != -1):
            locations_3.append(review.subratings.location)
        if(review.subratings.cleanliness != -1):
            cleanliness_3.append(review.subratings.cleanliness)
        if review.hotel.price != -1 :
            prices_3.append(review.hotel.price)
        if review.author == 'A TripAdvisor Member' :
            A_TripAdvisor_member_counter_3 += 1
        if review.author == 'lass=' :
            Lass_counter_3 += 1
        if not(all(ord(char) < 128 for char in review.content)):   
            non_ASCII_characters_3 += 1
            
    elif review.rating==4:
        reviews_4.append(review)
        content_lenghts_4.append(len(review.content))
        if(review.subratings.location != -1):
            locations_4.append(review.subratings.location)
        if(review.subratings.cleanliness != -1):
            cleanliness_4.append(review.subratings.cleanliness)
        if review.hotel.price != -1 :
            prices_4.append(review.hotel.price)
        if review.author == 'A TripAdvisor Member' :
            A_TripAdvisor_member_counter_4 += 1
        if review.author == 'lass=' :
            Lass_counter_4 += 1
        if not(all(ord(char) < 128 for char in review.content)):   
            non_ASCII_characters_4 += 1
               
    elif review.rating==5:
        reviews_5.append(review)
        content_lenghts_5.append(len(review.content))
        if(review.subratings.location != -1):
            locations_5.append(review.subratings.location)
        if(review.subratings.cleanliness != -1):
            cleanliness_5.append(review.subratings.cleanliness)
        if review.hotel.price != -1 :
            prices_5.append(review.hotel.price)
        if review.author == 'A TripAdvisor Member' :
            A_TripAdvisor_member_counter_5 += 1
        if review.author == 'lass=' :
            Lass_counter_5 += 1
        if not(all(ord(char) < 128 for char in review.content)):   
            non_ASCII_characters_5 += 1
        
reviews_dict_languages_1 = split_by_language(reviews_1)
reviews_dict_languages_2 = split_by_language(reviews_2)
reviews_dict_languages_3 = split_by_language(reviews_3)
reviews_dict_languages_4 = split_by_language(reviews_4)
reviews_dict_languages_5 = split_by_language(reviews_5)

n_fr_1 = len(reviews_dict_languages_1['fr'])  # number of french reviews whose rating is 1
n_en_1 = len(reviews_dict_languages_1['en'])  # number of english reviews whose rating is 1
n_de_1 = len(reviews_dict_languages_1['de'])
n_it_1 = len(reviews_dict_languages_1['it'])
n_es_1 = len(reviews_dict_languages_1['es'])
n_zh_1 = len(reviews_dict_languages_1['zh'])

n_fr_2 = len(reviews_dict_languages_2['fr'])  # number of french reviews whose rating is 2
n_en_2 = len(reviews_dict_languages_2['en'])  # number of english reviews whose rating is 2
n_de_2 = len(reviews_dict_languages_2['de'])
n_it_2 = len(reviews_dict_languages_2['it'])
n_es_2 = len(reviews_dict_languages_2['es'])
n_zh_2 = len(reviews_dict_languages_2['zh'])

n_fr_3 = len(reviews_dict_languages_3['fr'])
n_en_3 = len(reviews_dict_languages_3['en'])
n_de_3 = len(reviews_dict_languages_3['de'])
n_it_3 = len(reviews_dict_languages_3['it'])
n_es_3 = len(reviews_dict_languages_3['es'])
n_zh_3 = len(reviews_dict_languages_3['zh'])

n_fr_4 = len(reviews_dict_languages_4['fr'])
n_en_4 = len(reviews_dict_languages_4['en'])
n_de_4 = len(reviews_dict_languages_4['de'])
n_it_4 = len(reviews_dict_languages_4['it'])
n_es_4 = len(reviews_dict_languages_4['es'])
n_zh_4 = len(reviews_dict_languages_4['zh'])

n_fr_5 = len(reviews_dict_languages_5['fr'])
n_en_5 = len(reviews_dict_languages_5['en'])
n_de_5 = len(reviews_dict_languages_5['de'])
n_it_5 = len(reviews_dict_languages_5['it'])
n_es_5 = len(reviews_dict_languages_5['es'])
n_zh_5 = len(reviews_dict_languages_5['zh'])

n_fr = n_fr_1 + n_fr_2 + n_fr_3 + n_fr_4 + n_fr_5  # total number of french reviews
n_en = n_en_1 + n_en_2 + n_en_3 + n_en_4 + n_en_5  # total number of english reviews
n_de = n_de_1 + n_de_2 + n_de_3 + n_de_4 + n_de_5
n_it = n_it_1 + n_it_2 + n_it_3 + n_it_4 + n_it_5
n_es = n_es_1 + n_es_2 + n_es_3 + n_es_4 + n_es_5


print 'Percentage of French reviews whose rating is 1: ' + str(float(n_fr_1)/n_fr*100)
print 'Percentage of English reviews whose rating is 1: ' + str(float(n_en_1)/n_en*100)
print 'Percentage of German reviews whose rating is 1: ' + str(float(n_de_1)/n_de*100)
print 'Percentage of Italian reviews whose rating is 1: ' + str(float(n_it_1)/n_it*100)
print 'Percentage of Spanish reviews whose rating is 1: ' + str(float(n_es_1)/n_es*100)

print
print 'Percentage of French reviews whose rating is 2: ' + str(float(n_fr_2)/n_fr*100)
print 'Percentage of English reviews whose rating is 2: ' + str(float(n_en_2)/n_en*100)
print 'Percentage of German reviews whose rating is 2: ' + str(float(n_de_2)/n_de*100)
print 'Percentage of Italian reviews whose rating is 2: ' + str(float(n_it_2)/n_it*100)
print 'Percentage of Spanish reviews whose rating is 2: ' + str(float(n_es_2)/n_es*100)

print
print 'Percentage of French reviews whose rating is 3: ' + str(float(n_fr_3)/n_fr*100)
print 'Percentage of English reviews whose rating is 3: ' + str(float(n_en_3)/n_en*100)
print 'Percentage of German reviews whose rating is 3: ' + str(float(n_de_3)/n_de*100)
print 'Percentage of Italian reviews whose rating is 3: ' + str(float(n_it_3)/n_it*100)
print 'Percentage of Spanish reviews whose rating is 3: ' + str(float(n_es_3)/n_es*100)

print
print 'Percentage of French reviews whose rating is 4: ' + str(float(n_fr_4)/n_fr*100)
print 'Percentage of English reviews whose rating is 4: ' + str(float(n_en_4)/n_en*100)
print 'Percentage of German reviews whose rating is 4: ' + str(float(n_de_4)/n_de*100)
print 'Percentage of Italian reviews whose rating is 4: ' + str(float(n_it_4)/n_it*100)
print 'Percentage of Spanish reviews whose rating is 4: ' + str(float(n_es_4)/n_es*100)

print
print 'Percentage of French reviews whose rating is 5: ' + str(float(n_fr_5)/n_fr*100)
print 'Percentage of English reviews whose rating is 5: ' + str(float(n_en_5)/n_en*100)
print 'Percentage of German reviews whose rating is 5: ' + str(float(n_de_5)/n_de*100)
print 'Percentage of Italian reviews whose rating is 5: ' + str(float(n_it_5)/n_it*100)
print 'Percentage of Spanish reviews whose rating is 5: ' + str(float(n_es_5)/n_es*100)


# filtering the reviews whose lenght is less than 4000
plt.hist(content_lenghts_1, bins=range(0,4000,100) , color ='b' , label='rating_1' , alpha=0.2 , normed=True)
plt.hist(content_lenghts_2, bins=range(0,4000,100) , color ='r' , label='rating_2' , alpha=0.3 , normed=True)
plt.hist(content_lenghts_3, bins=range(0,4000,100) , color ='k' , label='rating_3' , alpha=0.2 , normed=True)
plt.hist(content_lenghts_4, bins=range(0,4000,100) , color ='y' , label='rating_4' , alpha=0.4 , normed=True)
plt.hist(content_lenghts_5, bins=range(0,4000,100) , color ='b' , label='rating_5' , alpha=0.1 , normed=True)
plt.title("Histogram for the content lengths of the reviews")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.legend()
plt.show()


print
print 'Percentage of the rated 1 reviews which contain non-ASCII characters: ' + str(float(non_ASCII_characters_1)/len(reviews_1))   
print 'Percentage of the rated 2 reviews which contain non-ASCII characters: ' + str(float(non_ASCII_characters_2)/len(reviews_2))
print 'Percentage of the rated 3 reviews which contain non-ASCII characters: ' + str(float(non_ASCII_characters_3)/len(reviews_3))
print 'Percentage of the rated 4 reviews which contain non-ASCII characters: ' + str(float(non_ASCII_characters_4)/len(reviews_4))
print 'Percentage of the rated 5 reviews which contain non-ASCII characters: ' + str(float(non_ASCII_characters_5)/len(reviews_5))


plt.hist(prices_1, bins=range(0,1000,50) , color ='b' , label='rating_1' , alpha=0.2 , normed=True)
plt.hist(prices_2, bins=range(0,1000,50) , color ='r' , label='rating_2' , alpha=0.3 , normed=True)
plt.hist(prices_3, bins=range(0,1000,50) , color ='k' , label='rating_3' , alpha=0.2 , normed=True)
plt.hist(prices_4, bins=range(0,1000,50) , color ='y' , label='rating_4' , alpha=0.4 , normed=True)
plt.hist(prices_5, bins=range(0,1000,50) , color ='b' , label='rating_5' , alpha=0.1 , normed=True)
plt.title("Histogram for the hotel prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.legend()
plt.show()

