# -*- coding: utf-8 -*-
"""
WeFarm Data Topic Modelling First Pass

@author: mike.a.taylor@ accenture.com (adapted from Olivier Grisel, Lars Buitinck & Chyi-Kwei Yau)

Todos
1. Tweak stop word removal
2. stemming/lemmatisation improvements e.g. lemmatise THEN stem as well
3. seperate the languag treatment (i.e. model topics for swahili and english seperately)
4. tune LDA/NMF, including test/train perplexity, regularisation, topic word manual quality checks, number of topics etc.
5. I'm sure there's more, comparison/augmentation with word2vec perhaps?
6. could be sped up by not using the duplcated conversations

"""

##############################################################################
# Library Imports
##############################################################################

### generic functions/data manipulation
from __future__ import print_function
from time import time
import numpy as np # needed for the array manipulation at the end
import pandas as pd

### sklearn for NLP pipeline and topic modelling tools
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction import text

### nltk for additional pipeline activities (stemming/lematisation etc.)
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

##############################################################################
# Custom Functions
##############################################################################

### Stemmer (feeds tokenise function)
stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    if lemmatise is True:
        for item in tokens:
            stemmed.append(wordnet_lemmatizer.lemmatize(item))
    else:        
        for item in tokens:
            stemmed.append(stemmer.stem(item))
    return stemmed

### Custom Tokeniser (slightly inefficient as uses python natively)
import string
def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

### Print the topics
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("--------------- " + "Topic #%d:" % topic_idx + " ---------------")
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
              + ' | ' for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print("")

### Create a dataframe of the topic's words and relative support
def df_top_words(model, feature_names, n_top_words):
    df = pd.DataFrame(columns = ["Topic","Ngram", "Support"])                                  
    for topic_idx, topic in enumerate(model.components_):
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            df = df.append({"Topic" : topic_idx,
                            "Ngram" : feature_names[i],
                            "Support": str(round(topic[i], 2))},
                            ignore_index = True)
    return df

def concatConv(df):
    df_tmp = df
    df_tmp = df_tmp.rename(columns={'body': 'Conversation'})
    df_Conv = df_tmp[["thread_id","Conversation"]].groupby(["thread_id"])["Conversation"].transform(lambda x: ' | '.join(x))
    df_Conv_Enriched = pd.concat([df, df_Conv], axis=1, join_axes=[datasetPD.index])
    df_Conv_Enriched["Conversation Answers"] = df_Conv_Enriched["Conversation"].apply(lambda x: x.count('|'))
    return df_Conv_Enriched
    
##############################################################################
# Constants etc.
##############################################################################

### collapse conversations (i.e. filter on "Q"? If you do you lose the view of the answer-user topic info, 
### but get a smaller data set
collapseConv = False

### make topic modelling agnostic? (i.e. do you filter on langauges?  to be simple I didn't but 
### catering for this would improve the model)
langAgnostic = True

### Ngram Ranges
mingram,maxgram = 1,2

### LDA/NMF inputs    
n_features = 100
n_topics = 15
n_top_words = 4 

### Extended stopwords
weFarmStopWords = ["farm","wefarm","yes","need","want","know","use",
                   "farming","farmer","to","the","can","and","is",
                   "does", "just", "like", "ur", "na", "ya", "make","dont"]
stopWords = text.ENGLISH_STOP_WORDS.union(weFarmStopWords)

lemmatise = True #(wordnet_lemmatizer used, if false it stems using Porter, could combine I reckon)

##############################################################################
# Main Code
##############################################################################

###Import and scrub data
print("Loading dataset...")
t0 = time()
#import (data location in same folder as this code please :))
datasetPD = pd.read_csv(r"threaded-data-with-langdetect.csv", encoding = "ISO-8859-1")

### Concatenate Conversations (NB this is worht doing regardles of collapseConv as it creates a richer rep of data)
#datasetPD_tmp = datasetPD
#datasetPD_tmp = datasetPD_tmp.rename(columns={'body': 'Conversation'})
#datasetPD_Conv = datasetPD_tmp[["thread_id","Conversation"]].groupby(["thread_id"])["Conversation"].transform(lambda x: ' | '.join(x))
#datasetPD_Conv_Enriched = pd.concat([datasetPD, datasetPD_Conv], axis=1, join_axes=[datasetPD.index])
datasetPD_Conv_Enriched = concatConv(datasetPD)

### DF filters
if collapseConv == True:
    datasetPD_Conv_Enriched = datasetPD_Conv_Enriched[(datasetPD['type'] =="Q")]
if langAgnostic == False:
    datasetPD_Conv_Enriched = datasetPD_Conv_Enriched[((datasetPD['lang'] =="EN") & 
                                                       (datasetPD['Detected_Language'] =="en"))]
                                                       
datasetPD_Conv_Enriched = datasetPD_Conv_Enriched.reset_index()
datasetLst = datasetPD_Conv_Enriched["Conversation"].tolist()
data_samples = datasetLst
print("English Records in data set - ", len(data_samples))
print("done in %0.3fs." % (time() - t0))

### descriptive stats/constants
n_samples = len(data_samples)

print("")
print("Topic Modelling w/ LDA and NMF")
print("")

### Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...") 
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize,
                                   max_df = 0.95,
                                   min_df = 2,
                                   max_features = n_features,
                                   stop_words = stopWords,
                                   lowercase = True,
                                   ngram_range = (mingram,maxgram))                                   
                    
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

### Fit the NMF model
print("Fitting the NMF model with tf-idf features,"
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_topics,
          random_state=1,
          alpha=.1,
          l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

### Print and create NMF Topic tables
print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)
NMFtopicTbl = df_top_words(nmf, tfidf_feature_names, n_top_words)

print ("---------------------------------------------------------------")
print ("---------------------------------------------------------------")
print ("")

### Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(tokenizer=tokenize,
                                max_df = 0.95,
                                min_df = 2,
                                max_features = n_features,
                                stop_words = stopWords,
                                lowercase = True,
                                ngram_range = (mingram,maxgram))
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics,
                                max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

### Print and create LDA Topic tables
print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
LDAtopicTbl = df_top_words(lda, tf_feature_names, n_top_words)

print ("---------------------------------------------------------------")
print ("---------------------------------------------------------------")
print ("")
print("Topic Modelling Complete, saving results...")

### Update the original dataframe with the degree of topic support per topic + the final selected dominant topic
### predict topics
LDAtf_new = lda.transform(tf)
for i in range(0,n_topics):
    Title = "LDA Topic Support " + str(i)
    datasetPD_Conv_Enriched[Title] = LDAtf_new[:,i] # utlises numpy array manipulation (falls/fell over without np import)
datasetPD_Conv_Enriched["LDA Main Topic"] = np.argmax(LDAtf_new,axis=1) # utlises numpy array manipulation (falls/fell over without np import)

NMFtf_new = lda.transform(tfidf)
for i in range(0,n_topics):
    Title = "NMF Topic Support " + str(i)
    datasetPD_Conv_Enriched[Title] = NMFtf_new[:,i] # utlises numpy array manipulation (falls/fell over without np import)
datasetPD_Conv_Enriched["NMF Main Topic"] = np.argmax(NMFtf_new,axis=1) # utlises numpy array manipulation (falls/fell over without np import)

### Save DFs to csvs 
datasetPD_Conv_Enriched.to_csv("threaded-data-with-langdetect+topics-EN+SW.csv")
LDAtopicTbl.to_csv("LDAtopic-table-SW.csv")
NMFtopicTbl.to_csv("NMFtopic-table-SW.csv")

