"""Script creating, saving and loading the data pickle file."""

import data


def main():
  data.create_pickled_data(overwrite_old=True)
  dataset = data.load_pickled_data()
  train_set = dataset['train']
  test_set = dataset['test']


if __name__ == '__main__':
  main()