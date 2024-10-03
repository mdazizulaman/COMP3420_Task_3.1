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
    # Reading the CSV file
  dataset = pd.read_csv(csv_file_path)

  # Extracting questions
  questions = dataset['qtext'].tolist()
  print(questions)

  # Combining questions into a single string
  unique_questions = set(questions)
  all_text = " ".join(unique_questions)

  # Tokenizing sentences
  sentences = sent_tokenize(all_text)
  # Create a list of lists for words in each sentence
  tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
  #Cleaning the tokens
  words = [word.lower() for sentence in tokenized_sentences for word in sentence if word.isalpha()]
  print(words)
  # Using NLTK's pos_tag_sents with Universal tagset for efficient tagging
  pos_tags = nltk.pos_tag_sents(tokenized_sentences, tagset='universal')

  # Counting occurrences of each noun
  noun_counts = collections.Counter()
  for tags in pos_tags:
     for word, tag  in tags:
        if tag =='NOUN':
           noun_counts[word] += 1

  # Getting the top N most common nouns
  top_n_nouns = noun_counts.most_common(N)
  #returning the result
  return top_n_nouns


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

