
def fix_zero_predictions(predictions, reviews):
    """
    If the context of a review contains less than ten percent ASCII symbols and its length lies between 41 and 52
    symbols, then it should be predicted as a zero rated review.

    :param predictions: The predictions made by the ensemble, which should be processed
    :param reviews: The dataset of reviews to be checked for the constraints
    :return: The altered predictions
    """
    for x in range(len(predictions)):
        if ascii_percentage(reviews[x]) < 0.1 and 41 < len(reviews[x]) < 52:
            predictions[x] = 0
    return predictions


def ascii_percentage(string):
    """
    For each character in the string, determine if it's an ASCII character and calculate the total ASCII character
    percentage of the string.

    :param string: Given string
    :return: ASCII percentage of the string
    """
    count = 0
    for char in string:
        if ord(char)<128:
            count += 1
    if len(string) == 0:
        return 1
    else:
        return float(count)/len(string)