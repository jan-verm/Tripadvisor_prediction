import data
from nltk.stem import SnowballStemmer
import langid
import numpy
import enchant
import csv
from textblob import TextBlob
from remove_stop_words import sanitize_2
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer
import unicodedata
import math
import sys

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

        # Print counter for visualizing progress
        if count%1000 ==1:
            print count
        count+=1

    return reviews_dict_languages


def correct_spelling_and_stem(language, review_list):
    """
    Remove accents and stopwords, correct the spelling, and apply stemming/lemmatization.
    input arguments:
        language: the language of the reviews to process
        review_list: a list of review items
    output arguments:
        return_list: a list of the reviews that have been preprocessed
    """

    # Initialize returnvalue
    return_list = []

    # Check if language is not Zh
    if language in language_map:

        # Load stopwords for the given language
        stopwords = get_stop_words(language)

        # The Eglish language gets a special threatment (lemmatization)
        if language == 'en':
            wordnet_lemmatizer = WordNetLemmatizer()

            # Use a counter to visualize the progress
            count = 1

            # Loop over available reviews
            for review in review_list:

                # Strip accents
                stripped_con = remove_diacritic(unicode(review.content.lower(), "utf-8"))

                # Tokenize (split words)
                con = TextBlob(stripped_con)
                con_token = list(con.words)

                # Remove stopwords
                con_token_stop = sanitize_2(con_token, stopwords)

                # Correct spelling, special case for english
                new_con = TextBlob(str(' '.join(con_token_stop)))
                corrected_con = new_con.correct()
                corrected_con_token = list(corrected_con.words)
                lemmatized_con = [ wordnet_lemmatizer.lemmatize(word) for word in corrected_con_token]

                # Make it one string again
                new_content = ' '.join(lemmatized_con)

                # Build new review and append to return list
                review1=data.Review(id=review.id, author=review.author, content=new_content, date=review.date, rating=review.rating,subratings=review.subratings, hotel=review.hotel)
                return_list.append(review1)

                # Print counter for visualizing progress
                print language +' '+ str(count)
                
                count+=1
        
        # Languages that are not En or Zh
        else:
            # Initialize stemmers and dictionaries
            stemmer = SnowballStemmer(language_map[language][1])
            enchant_dictionary = enchant.Dict(language_map[language][0])

            # Use a counter to visualize the progress
            count = 1

            # Loop over all available reviews
            for review in review_list:

                # Strip accents
                stripped_con = remove_diacritic(unicode(review.content.lower(), "utf-8"))

                # Tokenize (split words)
                tokenized_content = stripped_con.split()

                # Remove stopwords
                tokenized_content = sanitize_2(tokenized_content, stopwords)

                # Loop over all words within a review
                for x in range(0,len(tokenized_content)):

                    # Check if the word is spelled correctly
                    if enchant_dictionary.check(tokenized_content[x]) == False:

                        # Check if there are suggested corrections
                        if len(enchant_dictionary.suggest(tokenized_content[x])) > 0 :
                            try:
                                # Stem and correct the word
                                tokenized_content[x] = str(stemmer.stem(enchant_dictionary.suggest(tokenized_content[x])[0]))
                            except:
                                pass
                        else:
                            try:
                                # Stem the word
                                tokenized_content[x] = str(stemmer.stem(tokenized_content[x]))
                            except:
                                pass
                    # Words that were identified as written correct
                    else:
                        try:
                            # Stem the word
                            tokenized_content[x] = str(stemmer.stem(tokenized_content[x]))
                        except:
                            pass

                # Rejoin the tokenized words
                content = ' '.join(tokenized_content)

                # Build new review and append to return list
                return_list.append(data.Review(id=review.id, author=review.author, content=content, date=review.date, rating=review.rating,subratings=review.subratings, hotel=review.hotel))

                # Print counter for visualizing progress
                if count%1000 ==1:
                    print language +' '+ str(count)
                count+=1

    # Else clause handles Chinese only
    else:
        return_list = review_list

    return return_list


def save_reviews_to_csv(language, review_list, dataset):
    """
    Save the preprocessed reviews to a csv-file (one for each language).
    input arguments:
        language: the language of the reviews to process
        review_list: a list of review items
        dataset: either "train" or "test"
    output arguments:
        None
    """
    with open('reviews_'+dataset+'_'+language+'.csv', 'w') as csvfile:
        fieldnames = review_list[0].__dict__.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for review in review_list:
            writer.writerow(review.__dict__)


def remove_diacritic(input):
    """
    Accept a unicode string, and return a normal string without any diacritical marks.
    input arguments:
        input: the string to strip accents from
    output arguments:
        the stripped input
    """
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')

if __name__ == "__main__":
    dataset = sys.argv[1]
    if dataset == 'train':
        reviews = data.load_train()
    elif dataset == 'test':
        reviews = data.load_test()
    else:
        raise ValueError('No dataset ' + dataset + ' found!')
    print "reviews loaded"
    reviews_dict_languages = split_by_language(reviews)

    for k, v in reviews_dict_languages.iteritems():
        print k
        review_list = correct_spelling_and_stem(k, v)
        print "corrected and stemmed"
        save_reviews_to_csv(k, review_list, dataset)
        print "saved to csv"