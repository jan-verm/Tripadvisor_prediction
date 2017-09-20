"""Module for loading the train and test data.

This module supports loading the data in two seperate ways.
The first, which is done by calling the load_train and load_test functions,
opens and parses the text files one by one. Because of the abundance of files,
this can take a while.
The second way of loading the data is by opening a pickle file containing the
results of the load_train and load_test functions. To create this pickle file,
call the create_data_pickle function once. Afterwards, you will be ablo to
quickly load the data using the load_data_pickle function.
"""
import collections
import glob
import os
import re
import warnings

import utils


## Default data folder names ## 


DEFAULT_DATA_LOCATION = 'Data'
DEFAULT_TRAIN_DATA_LOCATION = os.path.join(DEFAULT_DATA_LOCATION, 'Train')
DEFAULT_TEST_DATA_LOCATION = os.path.join(DEFAULT_DATA_LOCATION, 'Test')

DEFAULT_PICKLE_PATH = os.path.join(DEFAULT_DATA_LOCATION, 'data.pkl')

PRICE_LOCATION_FILE_NAME = 'hotels.txt'
HOTEL_REVIEWS_FILE_TEMPLATE = 'Hotel_*.txt'
TEST_REVIEWS_FILE_TEMPLATE = '*.txt'


## Classes to store the parsed data in. ##


Hotel = collections.namedtuple(
  'Hotel', ['id', 'price', 'location'])
Review = collections.namedtuple(
  'Review', [
      'id', 'author', 'content', 'date', 'rating', 'subratings', 'hotel'])
Subratings = collections.namedtuple(
  'Subratings', [
      'value', 'rooms', 'location', 'cleanliness',
      'front_desk', 'service', 'business_service'])


## Helper functions for parsing data ##


def _parse_price_location_file(price_location_lines):
  hotels = {}
  for hotel_line in price_location_lines[1:]:
    tokens = hotel_line.strip().split(',')
    hotel_id = int(tokens[0])
    price = int(tokens[1])
    location = tokens[2]
    hotels[hotel_id] = Hotel(id=hotel_id, price=price, location=location)
  return hotels


def _extract_id_from_file_name(review_file_name):
  regex_res = re.findall('\d+', review_file_name)
  assert len(regex_res) == 1
  return int(regex_res[0])


_AUTHOR_TAG = '<Author>'
_CONTENT_TAG = '<Content>'
_DATE_TAG = '<Date>'
_RATING_TAG = '<Overall_rating>'
_SUBRATING_TAG = '<Subratings>'
_PRICE_TAG = '<Price>'
_LOCATION_TAG = '<Location>'


def _parse_string_line(line, tag):
  assert line.startswith(tag)
  return line.strip()[len(tag):]


def _parse_author_line(author_line):
  return _parse_string_line(author_line, _AUTHOR_TAG)


def _parse_content_line(content_line):
  return _parse_string_line(content_line, _CONTENT_TAG)


def _parse_date_line(date_line):
  return _parse_string_line(date_line, _DATE_TAG)


def _parse_rating_line(rating_line):
  rating_string = _parse_string_line(rating_line, _RATING_TAG)
  return int(rating_string)


def _parse_price_line(price_line):
  price_string = _parse_string_line(price_line, _PRICE_TAG)
  return int(price_string)


def _parse_location_line(location_line):
  return _parse_string_line(location_line, _LOCATION_TAG)


def _parse_subratings_line(subratings_line):
  subratings_string = _parse_string_line(subratings_line, _SUBRATING_TAG)
  subrating_tokens = subratings_string.split(',')

  subratings = {}
  for token in subrating_tokens:
    left, right = token.strip().split('=')
    subratings[left] = int(right)

  return Subratings(**subratings)


def _parse_single_hotel_review(review_lines, hotel):
  author_line, content_line, date_line, overall_rating_line, subratings_line = (
      review_lines)
  # Parse each of the lines
  author = _parse_author_line(author_line)
  content = _parse_content_line(content_line)
  date = _parse_date_line(date_line)
  rating = _parse_rating_line(overall_rating_line)
  subratings = _parse_subratings_line(subratings_line)
  return Review(
      id=-1, author=author, content=content, date=date, rating=rating,
      subratings=subratings, hotel=hotel)


def _parse_hotel_review_file(hotel_review_lines, hotel):
  assert len(hotel_review_lines)%6 == 0
  hotel_reviews = []
  for start_idx in xrange(0, len(hotel_review_lines), 6):
    review_lines = hotel_review_lines[start_idx:start_idx+5]
    hotel_reviews.append(_parse_single_hotel_review(review_lines, hotel))
  return hotel_reviews


def _parse_test_review_file(test_review_lines, review_id):
  assert len(test_review_lines) == 5
  author_line, content_line, date_line, price_line, location_line = (
      test_review_lines)
  author = _parse_author_line(author_line)
  content = _parse_content_line(content_line)
  date = _parse_date_line(date_line)
  price = _parse_price_line(price_line)
  location = _parse_location_line(location_line)
  unknown_hotel = Hotel(id=-1, price=price, location=location)
  review = Review(
      id=review_id, author=author, content=content, date=date, rating=-1,
      subratings=None, hotel=unknown_hotel)
  return review


## Functions for loading and parsing all of the data ##


def load_train(train_data_folder=DEFAULT_TRAIN_DATA_LOCATION):
  """Loads and parses the train data.

  Args:
    train_data_folder: string containing the path to the folder containing the
        training data.

  Returns:
    A list of all the reviews. Each review is a namedtuple object containing
    the author, content, date, rating, subratings and hotel data. The hotel
    data is a namedtuple containing the price and location of the hotel.
    The subratings is a namedtuple containing the value, rooms, location,
    cleanliness, front_desk, service and business_service ratings. Different
    reviews of the same hotel will point to the same Hotel object.
  """
  price_location_file_path = os.path.join(
      train_data_folder, PRICE_LOCATION_FILE_NAME)
  review_file_paths = glob.glob(os.path.join(
      train_data_folder, HOTEL_REVIEWS_FILE_TEMPLATE))
  
  # First, load and parse the hotel price locations file
  with open(price_location_file_path, 'rb') as price_location_file:
    price_location_lines = price_location_file.readlines()
  hotels_dict = _parse_price_location_file(price_location_lines)
  assert len(hotels_dict) == len(review_file_paths)
  
  # Second, load and parse each of the hotel review files
  reviews = []
  for review_file_path in review_file_paths:
    with open(review_file_path, 'rb') as review_file:
      review_file_lines = review_file.readlines()
    hotel_id = _extract_id_from_file_name(os.path.basename(review_file_path))
    hotel = hotels_dict[hotel_id]
    hotel_reviews = _parse_hotel_review_file(review_file_lines, hotel)
    reviews += hotel_reviews
  return reviews


def load_test(test_data_folder=DEFAULT_TEST_DATA_LOCATION):
  """Loads and parses the test data.

  Args:
    test_data_folder: string containing the path to the folder containing the]
        test data.

  Returns:
    A list of all the test reviews, sorted by the review id. Each review is a
    namedtuple object containing the author, content, date and hotel data. The
    hotel data is a namedtuple containing the price and location of the hotel.
    Since the hotel id is unknown, different reviews of the same hotel will 
    point to different Hotel objects.
  """
  review_file_paths = glob.glob(os.path.join(
      test_data_folder, TEST_REVIEWS_FILE_TEMPLATE))
  # Parse all the review files one by one
  reviews = []
  for review_file_path in review_file_paths:
    review_id = _extract_id_from_file_name(os.path.basename(review_file_path))
    with open(review_file_path, 'rb') as review_file:
      review_file_lines = review_file.readlines()
    reviews.append(_parse_test_review_file(review_file_lines, review_id))
  # Sort them by review id
  key_getter = lambda r: r.id
  reviews.sort(key=key_getter)
  assert range(1, len(reviews)+1) == map(key_getter, reviews)
  return reviews


## Functions for loading the data from a pickle file. ##


def create_pickled_data(train_data_folder=DEFAULT_TRAIN_DATA_LOCATION,
                        test_data_folder=DEFAULT_TEST_DATA_LOCATION,
                        pickled_data_file_path=DEFAULT_PICKLE_PATH,
                        overwrite_old=True):
  """Creates the data pickle file.

  Loads and parses the train and test data, and then writes it to a single
  pickle file.

  Args: 
    train_data_folder: path to the train data folder.
    test_data_folder: path to the test data folder.
    pickled_data_file_path: location where the resulting pickle file should
        be stored.
  """
  if os.path.exists(pickled_data_file_path):
    if not overwrite_old:
      return 
    warnings.warn(
        "There already exists a data pickle file, which will be overwritten.")
  train_data = load_train(train_data_folder)
  test_data = load_test(test_data_folder)
  utils.dump_pickle(
      dict(train=train_data, test=test_data), pickled_data_file_path)


def load_pickled_data(pickled_data_file_path=DEFAULT_PICKLE_PATH):
  """Loads the train and test data from a pickle file.

  Args:
    pickled_data_file_path: location of the data pickle file.
  """
  return utils.load_pickle(pickled_data_file_path)
