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
  # Reading CSV file
  dataset = pd.read_csv(csv_file_path)

  # Extracting questions
  questions = dataset['qtext'].tolist()
  #print(questions)

  # Combining questions into a single string
  unique_questions = set(questions)
  all_text = " ".join(unique_questions)

  # Tokenizing sentences
  sentences = sent_tokenize(all_text)
  # Create a list of lists for words in each sentence
  tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
  #Cleaning the tokens
  words = [word.lower() for sentence in tokenized_sentences for word in sentence if word.isalpha()]
  #print(words)
  
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
    # Reading CSV file
    dataset = pd.read_csv(csv_file_path)
    
    # Extracting unique questions
    unique_questions = set(dataset['qtext'].tolist())
    
    # Concatenating all unique questions into a single string
    all_text = " ".join(unique_questions)
    
    # Tokenizing the text into sentences and then into words for each sentence
    sentences = sent_tokenize(all_text)
    # Non-stemmed
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]  

    # Generating non-stemmed 2-grams manually
    non_stemmed_2grams = []
    for sentence in tokenized_sentences:
        for i in range(len(sentence) - 1):
            non_stemmed_2grams.append((sentence[i], sentence[i+1]))
    
    # Using Counter to count the occurrences of non-stemmed 2-grams
    non_stemmed_counts = collections.Counter(non_stemmed_2grams)
    
    # Normalizing frequencies by dividing by the total number of non-stemmed 2-grams
    total_non_stemmed = sum(non_stemmed_counts.values())
    non_stemmed_freqs = [(bigram, round(count / total_non_stemmed, 4)) for bigram, count in non_stemmed_counts.items()]
    
    # Sorting the 2-grams by frequency in descending order
    non_stemmed_freqs.sort(key=lambda x: x[1], reverse=True)
    
    # Again repeating the similar process for case-sensitive 2-grams
    tokenized_sentences_case_sensitive = [word_tokenize(sentence) for sentence in sentences]  

    # Generating case-sensitive 2-grams
    stemmed_2grams = []
    for sentence in tokenized_sentences_case_sensitive:
        for i in range(len(sentence) - 1):
            stemmed_2grams.append((sentence[i], sentence[i+1]))
    
    # Us ingCounter to count the occurrences of case-sensitive 2-grams
    stemmed_counts = collections.Counter(stemmed_2grams)
    
    # Normalizing frequencies by dividing by the total number of case-sensitive 2-grams
    total_stemmed = sum(stemmed_counts.values())
    stemmed_freqs = [(bigram, round(count / total_stemmed, 4)) for bigram, count in stemmed_counts.items()]
    
    # Sorting the 2-grams by frequency in descending order
    stemmed_freqs.sort(key=lambda x: x[1], reverse=True)
    
    # Returning the topN_2grams for both non-stemmed and stemmed 2-grams
    return non_stemmed_freqs[:N], stemmed_freqs[:N]
    

# Task 3 (5 marks)
def sim_tfidf(csv_file_path):
    # Loading data
    data = pd.read_csv(csv_file_path)
    
    # Getting unique questions and corresponding candidate sentences
    unique_questions = data['qtext'].unique()
    unique_answers = data['atext'].unique()
    
    # Combining unique questions and answers
    all_texts = np.concatenate((unique_questions, unique_answers))
    
    # Initializing TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the combined texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Preparing to calculate proportion of correctly answered questions
    correct_count = 0
    total_questions = 0
    
    # Iterating through each unique question
    for question in unique_questions:
        # Getting the corresponding candidate sentences
        question_data = data[data['qtext'] == question]
        candidate_sentences = question_data['atext'].values
        labels = question_data['label'].values
        
        # Skipping if there are no candidate sentences
        if len(candidate_sentences) == 0:
            continue
        
        # Preparing the input for tfidf_matrix
        input_texts = [question] + list(candidate_sentences)
        
        # Transforming the input texts to get their TF-IDF representation
        input_tfidf = vectorizer.transform(input_texts)
        
        # Calculating cosine similarities using matrix multiplication
        cosine_similarities = (input_tfidf[0] @ input_tfidf[1:].T).A[0]
        
        # Finding the index of the most similar sentence
        max_sim_index = np.argmax(cosine_similarities)
        
        # Checking if the most similar sentence has label 1
        actual_label = labels[max_sim_index]
        
        # Updating counts
        if actual_label == 1:
            correct_count += 1
        total_questions += 1
    
    # Calculate and return the proportion of correctly answered questions
    return round(correct_count / total_questions, 2) if total_questions > 0 else 0.0
    


# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    print(topN_pos('train.csv', 3))
    print("------------------------")
    print(topN_2grams('train.csv', 3))
    print("------------------------")
    print(sim_tfidf('train.csv'))

