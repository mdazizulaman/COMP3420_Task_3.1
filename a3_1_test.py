import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import pandas as pd

import unittest

import a3_1

class TestBasic(unittest.TestCase):
    def test_q1(self):
        
        result = a3_1.topN_pos('train.csv', 3)
        # print(result)
        self.assertTrue(result==[('cancer', 157), ('symptoms', 116), ('help', 87)]) 

    def test_q2(self):
        result = a3_1.topN_2grams('train.csv', 3)               
        
        self.assertTrue(result== ([(('what', 'are'), 0.0177), (('how', 'can'), 0.0126), (('what', 'is'), 0.0125)],
                                  [(('What', 'are'), 0.0174), (('How', 'can'), 0.0123), (('What', 'is'), 0.0122)]))

    
    def test_q3(self):
        result = a3_1.sim_tfidf('train.csv')
        self.assertEqual(result,0.54)

if __name__ == "__main__":
    unittest.main()
