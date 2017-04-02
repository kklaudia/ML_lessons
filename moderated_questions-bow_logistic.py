# -*- coding: utf-8 -*-
"""
Moderated answers
"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from math import log
from sklearn.model_selection import train_test_split
from datetime import datetime

plt.style.use('ggplot')

print("Loading data...")
#Read all the data
all_data = pd.read_csv("question_quality_classification.csv")

#Create a sample that contain 50:50 moderated and not moderated questions
ones = all_data[all_data["moderated_pos"]==1][45:]
zeros = all_data[all_data["moderated_pos"]==0]

zeros_sample = zeros.sample(n=106000)

new_df = pd.concat([ones, zeros_sample], axis=0)

#Use only sample of 50000 records (to make it faster)
data = new_df.sample(n=50000).reset_index(drop=True)

X = data["content"]
y = data["moderated_pos"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print("Preprocessing...")

def preprocessing(text):
    """Change text to lowercase, eliminate HTML tags, remove non-alpha numerical characters, substitute digits with <DIG>"""
    text = text.lower()
    
    cleanhtml = re.compile('<.*?>')
    cleantext = re.sub(cleanhtml, '', text)
    
    cleantext = re.sub('[^0-9a-zA-Z]+', " ", cleantext)
    cleantext = re.sub("\d+", " <DIG> ", cleantext)
    
    return cleantext

def remove_stop_words(text):
    """Remove stop words from text"""
    stop = set(stopwords.words('english'))
    text = text.lower()
    text_list = text.split(" ")
    
    new_text = [word for word in text_list if word not in stop]
    improved_text = " ".join(new_text)
    
    return improved_text

#Apply preprocessing
X_train = X_train.apply(preprocessing)
X_test = X_test.apply(preprocessing)
print("Removing stop words...")

X_train = X_train.apply(remove_stop_words)
X_test = X_test.apply(remove_stop_words)
print("Creating BOW dictionary...")

#Create bag of words dictionary

def build_dictionary(corpus):
    """Builds a dictionary with the most frequent words"""
    bag_of_words = {}
    for question in corpus:
        question = question.split(" ")
        for word in question:
            if len(word)>=3:
                if word in bag_of_words.keys():
                    bag_of_words[word] += 1
                else:
                    bag_of_words[word] = 1
    return bag_of_words

word_counts = pd.DataFrame.from_dict(build_dictionary(X_train), orient='index').reset_index()
word_counts = word_counts.rename(columns={"index": "word", 0:"occurances"}).sort_values(by="occurances", ascending=False)
word_counts.index = np.arange(1, len(word_counts) + 1)

word_counts_slice = word_counts[word_counts["occurances"]>=20]
bow_dict = word_counts_slice["word"].to_dict()
bow_dict_rev = {v: k for k, v in bow_dict.items()}

print("Building word vectors...")

#Build matrix of word vectors
def build_vectors(data_set, bow_dict_rev):
    """Creates a word vector for each question"""
    vectors = []
    
    for question in data_set:
        question = question.split(" ")
        vector = []
        for word in question:
            if word in bow_dict_rev.keys():
                id_code = bow_dict_rev[word]
                vector.append(id_code)
        vectors.append(vector)
    
    vectors = np.array(vectors)
    return vectors

vectors_matrix_train = build_vectors(X_train, bow_dict_rev)
vectors_matrix_test = build_vectors(X_test, bow_dict_rev)

#Logistic regression
print("Running logistic regression...")


def count_h(vectors_matrix, w_values):
    h_list = []
    for i in range(len(vectors_matrix)):
        z = w_values[0]
        for word_id in vectors_matrix[i]:
            z += w_values[word_id]
        h = 1 / (1 + np.exp(-z))
        h_list.append(h)
    return h_list

def count_error(h_list, y):
    m = len(y)
    error = 0
    for i in range(len(y)):
        if y.iloc[i] == 1:
            error += -log(h_list[i])
        else:
            error += -log(1-h_list[i])
    error = error/m
    return error


def logistic_regression(vectors_matrix, y, bow_dict, alpha, iterations, error_value = 0.000001):
    w_values = np.zeros(len(bow_dict)+1)
    vectors_matrix = np.array([[0] + vector for vector in vectors_matrix])
    
    m = len(vectors_matrix)
    e = 0
    error_records = []
    
    for j in range(iterations):
        
        h = []
        loss = []
        gradient = [0] * len(w_values)
        for i in range(len(vectors_matrix)):
            z = w_values[0]
            
            for word_id in vectors_matrix[i]:
                z += w_values[word_id]
                
            h_partial = 1 / (1 + np.exp(-z))
            h.append(h_partial)
            loss_partial = h_partial - y.iloc[i]
            loss.append(loss_partial)
            
            for word_id in vectors_matrix[i]:
                gradient[word_id] += loss_partial        
        
        temp = np.zeros(len(w_values))
        
        for w in range(len(w_values)):
            temp[w] = w_values[w] - ((alpha * gradient[w]) / m)
            
        w_values = temp

        error = count_error(h, y)
        
        if j % 10 == 0:
            error_records.append(error)
        
        if abs(e - error) > error_value:
            e = error
        else:
            break
    
    count_h(vectors_matrix, w_values)
    error = count_error(h, y)
    error_records.append(error)
    
    return w_values, error_records


alpha = 0.05
iterations = 300
error_value = 0.000001

t1 = datetime.today()
w_values_train, error_records_train = logistic_regression(vectors_matrix_train, y_train, bow_dict, alpha, iterations, error_value)
w_values_test, error_records_test = logistic_regression(vectors_matrix_test, y_test, bow_dict, alpha, iterations, error_value)
t2 = datetime.today()
print("Training time: ", (t2-t1))

iterations_count = list(range(len(error_records_train)))

plt.figure()
plt.plot(iterations_count, error_records_train)
plt.plot(iterations_count, error_records_test)
plt.title("Error vs iterations")
plt.show()

print("TRAIN error: %s" % error_records_train[-1])
print("TEST error: %s" % error_records_test[-1])

h_train = count_h(vectors_matrix_train, w_values_train)
h_test = count_h(vectors_matrix_test, w_values_test)

def accuracy(h):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for i in range(len(h)):
        if h[i] >= 0.5 and y.iloc[i]:
            tp += 1
        elif h[i] < 0.5 and y.iloc[i]:
            fn += 1
        elif h[i] >= 0.5 and not y.iloc[i]:
            fp += 1
        elif h[i] < 0.5 and not y.iloc[i]:
            tn += 1
            
    train_sum = tp + tn + fp + fn
    print("Accuracy: %s" % ((tp + tn) * 100/train_sum))
    print("Precision: %s" % (tp*100/(tp+fp)))
    print("Recall: %s" % (tp*100/(tp+fn)))
    return tp, tn, fp, fn

print("TRAIN")
tp1, tn1, fp1, fn1 = accuracy(h_train)

print("\nTEST")
tp2, tn2, fp2, fn2 = accuracy(h_test)

plt.show()


