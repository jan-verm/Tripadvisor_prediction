"""Simple script for generating Kaggle compatible test set predictions.

Usage: python create_submission <path_to_predictions.npy> <path_to_output.csv>
"""
import csv
import os
import sys

import utils


def write_predictions_to_csv(predictions, out_path):
  """Writes the predictions to a csv file.

  Assumes the predictions are ordered by review id.
  """
  with open(out_path, 'wb') as outfile:
    # Initialise the writer
    csvwriter = csv.writer(
        outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # Write the header
    csvwriter.writerow(['id', 'rating'])
    # Write the rows using 18 digit precision
    for idx, prediction in enumerate(predictions):
      csvwriter.writerow([str(idx+1), "%.18f" % prediction])


def main(in_path, out_path):
  predictions = utils.load_npy(in_path)  
  write_predictions_to_csv(predictions, out_path)
  print 'Generated predictions file %s' % out_path


def _parse_args(args):
  if not len(args) in (2, 3):
    print ('Usage: python create_submission <path_to_predictions.npy> '
           '[<path_to_output.csv>]')
    return
  in_path = sys.argv[1]
  out_path = sys.argv[2] if len(sys.argv) == 3 else in_path+'.csv'
  return in_path, out_path


if __name__ == '__main__':
  args = sys.argv
  arguments = _parse_args(args)
  if arguments:
    main(*arguments)