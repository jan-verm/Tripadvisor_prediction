"""Module containing utility functions."""
import os
import time

try:
  import cPickle as pickle
except:
  print "Warning: Couldn't import cPickle, using native pickle instead."
  import pickle

import numpy as np


## Disk access function ##

def _make_dir(path):
  path_dir = os.path.dirname(path)
  if not os.path.exists(path_dir):
    os.makedirs(path_dir)
    

def dump_pickle(data, path):
  """Dumps data to a pkl file."""
  if not path.endswith('.pkl'):
    raise ValueError(
        'Pickle files should end with .pkl, but got %s instead' % path)
  _make_dir(path)
  with open(path, 'wb') as pkl_file:
    pickle.dump(data, pkl_file, pickle.HIGHEST_PROTOCOL)


def load_pickle(path_to_pickle):
  with open(path_to_pickle, 'rb') as pkl_file:
    return pickle.load(pkl_file)


def dump_npy(array, path):
  """Dumps a single numpy array to a npy file."""
  if not path.endswith('.npy'):
    raise ValueError(
        'Filename should end with .npy, but got %s instead' % path)
  _make_dir(path)
  with open(path, 'wb') as npy_file:
    np.save(npy_file, array)


def load_npy(path):
  return np.load(path)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def generate_unqiue_file_name(basename, file_ext):
  return basename + '_' + timestamp() + '.' + file_ext