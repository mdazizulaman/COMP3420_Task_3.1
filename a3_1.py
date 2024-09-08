import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import pandas as pd

# Task 1 (5 marks)
def topN_pos(csv_file_path, N):
    """
    Example:
    >>> topN_pos('train.csv', 3)
    output would look like [(noun1, 22), (noun2, 10), ...]
    """


# Task 2 (5 marks)
def topN_2grams(csv_file_path, N):
    """
    Example:
    >>> topN_2grams('train.csv', 3)
    output would look like [('what', 'is', 0.4113), ('how', 'many', 0.2139), ....], [('I', 'feel', 0.1264), ('pain', 'in', 0.2132), ...]
    """
    
    

# Task 3 (5 marks)
def sim_tfidf(csv_file_path):
    """
    Example:
    >>> sim_tfidf('train.csv')
    output format would be like 0.54
   
    """
    


# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    print(topN_pos('train.csv', 3))
    print("------------------------")
    print(topN_2grams('train.csv', 3))
    print("------------------------")
    print(sim_tfidf('train.csv'))

