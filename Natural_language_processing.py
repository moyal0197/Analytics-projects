#!/usr/bin/env python
# coding: utf-8

# # Text Mining using Natural Language Processing (NLP)

# ## Part 1: Reading in the Yelp Reviews

# - "corpus" = collection of documents
# - "corpora" = plural form of corpus

# In[1]:


#import required packages
#basics
import pandas as pd 
import numpy as np

#misc
import gc
import time
import warnings

#stats
#from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss

#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
#import matplotlib_venn as venn

#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords

#import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize

# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   


#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, decomposition, ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import  textblob
#import xgboost
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

from textblob import TextBlob
from nltk.stem import PorterStemmer
import nltk
#nltk.download('wordnet')
from textblob import Word

#settings
start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# read yelp.csv into a DataFrame
yelp = pd.read_csv('C:\\Users\\ADMIN\\Desktop\\Analytix Labs\\PYTHON\\Text Mining\\yelp.csv')


# In[3]:


yelp.head(5)


# In[4]:


yelp=yelp[['review_id', 'stars', 'text', 'cool', 'useful', 'funny']]


# In[5]:


yelp.head()


# In[7]:


df = yelp


# In[ ]:


df['text'] = df['text'].astype(str)
df['count_sent']=df["text"].apply(lambda x: len(re.findall("\n",str(x)))+1)

#Word count in each comment:
df['count_word']=df["text"].apply(lambda x: len(str(x).split()))

#Unique word count
df['count_unique_word']=df["text"].apply(lambda x: len(set(str(x).split())))

#Letter count
df['count_letters']=df["text"].apply(lambda x: len(str(x)))

#Word density

df['word_density'] = df['count_letters'] / (df['count_word']+1)

#punctuation count
df["count_punctuations"] =df["text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

#upper case words count
df["count_words_upper"] = df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

#upper case words count
df["count_words_lower"] = df["text"].apply(lambda x: len([w for w in str(x).split() if w.islower()]))

#title case words count
df["count_words_title"] = df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

#Number of stopwords
df["count_stopwords"] = df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

#Average length of the words
df["mean_word_len"] = df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

#Number of numeric
df['numeric'] = df['text'].apply(lambda x :len([x for x in x.split() if x.isdigit()]))

#Number of alphanumeric
df['alphanumeric'] = df['text'].apply(lambda x :len([x for x in x.split() if x.isalnum()]))

#Number of alphabetics
df['alphabetetics'] = df['text'].apply(lambda x :len([x for x in x.split() if x.isalpha()]))

#Number of alphabetics
df['Spaces'] = df['text'].apply(lambda x :len([x for x in x.split() if x.isspace()]))

#Number of Words ends with
df['words_ends_with_et'] = df['text'].apply(lambda x :len([x for x in x.lower().split() if x.endswith('et')]))

#Number of Words ends with
df['words_start_with_no'] = df['text'].apply(lambda x :len([x for x in x.lower().split() if x.startswith('no')]))

# Count the occurences of all words
df['wordcounts'] = df['text'].apply(lambda x :dict([ [t, x.split().count(t)] for t in set(x.split()) ]))

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

df['noun_count'] = df['text'].apply(lambda x: check_pos_tag(x, 'noun'))
df['verb_count'] = df['text'].apply(lambda x: check_pos_tag(x, 'verb'))
df['adj_count']  = df['text'].apply(lambda x: check_pos_tag(x, 'adj'))
df['adv_count']  = df['text'].apply(lambda x: check_pos_tag(x, 'adv'))
df['pron_count'] = df['text'].apply(lambda x: check_pos_tag(x, 'pron')) 


# In[8]:


df['sentiment'] = df["text"].apply(lambda x: TextBlob(x).sentiment.polarity )


# In[9]:


df.head()


# In[10]:


yelp.stars.value_counts()


# In[11]:


# create a new DataFrame that only contains the 5-star and 1-star reviews
#yelp_best_worst = yelp[(yelp.stars==5) | (yelp.stars==1)]

# define X and y
X = yelp.text
y = yelp.stars

# split the new DataFrame into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[11]:


yelp.shape


# In[ ]:


yelp.head()


# In[16]:


s = 'Analytixlabs is from bagnalore, it has offices in Gurgoan, KL. It is staarted in 4Years back'


# In[17]:


#Abbrevations and Words correction
def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r' +', ' ', text)
    text = re.sub(r"[-()\"#/@;:{}`+=~|.!?,'0-9]", "", text)
    return(text)


# In[18]:


clean_text(s)


# In[19]:


stop = set(nltk.corpus.stopwords.words('english'))


# In[21]:


print(stop)


# In[22]:


stemmer_func = nltk.stem.snowball.SnowballStemmer("english").stem


# In[25]:


s= 'Analytics is really doing good'


# In[30]:


stemmer_func('really')


# In[32]:


s.split()


# In[34]:


import string
def pre_process(text):
    #text = text.str.replace('/','')
    #text = text.apply(lambda x: re.sub("  "," ", x))
    #text = re.sub(r"[-()\"#/@;:{}`+=~|.!?,']", "", text)
    #text = re.sub(r'[0-9]+', '', text)
    #text = text.apply(lambda x: " ".join(x.translate(str.maketrans('', '', string.punctuation)) for x in x.split() if x.isalpha()))
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    #text = text.apply(lambda x: str(TextBlob(x).correct()))
    #text = text.apply(lambda x: " ".join(PorterStemmer().stem(word) for word in x.split()))
    #text = text.apply(lambda x: " ".join(stemmer_func(word) for word in x.split()))
    #text = text.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    #text = text.apply(lambda x: " ".join(word for word, pos in pos_tag(x.split()) if pos not in ['NN','NNS','NNP','NNPS']))
    return(text)


# In[ ]:





# In[35]:


X_train = X_train.apply(lambda x: clean_text(x))
X_test = X_test.apply(lambda x: clean_text(x))


# In[36]:


X_train=pre_process(X_train)
X_test=pre_process(X_test)


# In[ ]:


#Vectorization


# In[37]:


get_ipython().run_line_magic('pinfo', 'CountVectorizer')


# In[38]:


#Train
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1 ), min_df=5, encoding='latin-1' , max_features=800)
xtrain_count = count_vect.fit_transform(X_train)


# In[41]:


dtm=xtrain_count.toarray()


# In[45]:


dtm


# In[51]:


count_vect.get_feature_names()


# In[52]:


dtm1=pd.DataFrame(dtm)


# In[53]:


dtm1.columns=count_vect.get_feature_names()


# In[54]:


dtm1.head()


# In[55]:


#Train
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1 ), min_df=5, encoding='latin-1' , max_features=800)
xtrain_count = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(xtrain_count)

#Test
#count_vect = CountVectorizer()
xtest_count = count_vect.transform(X_test)

#tfidf_transformer = TfidfTransformer()
X_test_tfidf = tfidf_transformer.transform(xtest_count)


# In[56]:


dtm2=pd.DataFrame(X_train_tfidf.toarray(), columns=count_vect.get_feature_names())


# In[57]:


dtm2.head(100)


# In[59]:


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), max_features=800)
tfidf_vect_ngram.fit(df['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)
xtest_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)


# In[60]:


# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=800)
tfidf_vect_ngram_chars.fit(df['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_train) 
xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_test)


# In[61]:


#Topic Models as features

# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(X_train_tfidf)
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()


# In[63]:


# view the topic models
n_top_words = 50
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))

topic_summaries


# In[64]:


frequency_words_wo_stop= {}
for data in yelp['text']:
    tokens = nltk.wordpunct_tokenize(data.lower())
    for token in tokens:
        if token.lower() not in stop:
            if token in frequency_words_wo_stop:
                count = frequency_words_wo_stop[token]
                count = count + 1
                frequency_words_wo_stop[token] = count
            else:
                frequency_words_wo_stop[token] = 1
                


# In[65]:


frequency_words_wo_stop


# In[1]:


var = "chandr mouli rajesh rree chandra chandra mouli mouli rajesh rajesh"


# In[3]:


from wordcloud import WordCloud ,STOPWORDS


# In[4]:


wordcloud = WordCloud(stopwords=[]).generate(str(var.tolist()))
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(200,100))
plt.imshow(wordcloud)


# In[80]:


def train_model(classifier, feature_vector_train, label, feature_vector_valid,  valid_y, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


# In[81]:


#Naive Bayes
# Naive Bayes on Count Vectors and TF-IDF
accuracy_L1 = train_model(naive_bayes.MultinomialNB(), X_train_tfidf, y_train, X_test_tfidf, y_test)
print("NB  for L1, Count Vectors: ", accuracy_L1)



# Naive Bayes on Word Level TF IDF Vectors
accuracy_L1 = train_model(naive_bayes.MultinomialNB(), xtrain_count, y_train, xtest_count, y_test)
print("NB  for L1, WordLevel TF-IDF: ", accuracy_L1)



# Naive Bayes on Ngram Level TF IDF Vectors
accuracy_L1 = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram, y_test)
print("NB  for L1, N-Gram Vectors: ", accuracy_L1)



# Naive Bayes on Character Level TF IDF Vectors
accuracy_L1 = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, y_train, xtest_tfidf_ngram_chars, y_test)
print("NB for L1, CharLevel Vectors: ", accuracy_L1)


# In[83]:


#Naive Bayes
# Naive Bayes on Count Vectors and TF-IDF
accuracy_L1 = train_model(LogisticRegression(), X_train_tfidf, y_train, X_test_tfidf, y_test)
print("LR  for L1, Count Vectors: ", accuracy_L1)



# Naive Bayes on Word Level TF IDF Vectors
accuracy_L1 = train_model(LogisticRegression(), xtrain_count, y_train, xtest_count, y_test)
print("LR  for L1, WordLevel TF-IDF: ", accuracy_L1)



# Naive Bayes on Ngram Level TF IDF Vectors
accuracy_L1 = train_model(LogisticRegression(), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram, y_test)
print("LR  for L1, N-Gram Vectors: ", accuracy_L1)



# Naive Bayes on Character Level TF IDF Vectors
accuracy_L1 = train_model(LogisticRegression(), xtrain_tfidf_ngram_chars, y_train, xtest_tfidf_ngram_chars, y_test)
print("LR for L1, CharLevel Vectors: ", accuracy_L1)


# In[85]:


#Naive Bayes
# Naive Bayes on Count Vectors and TF-IDF
accuracy_L1 = train_model(svm.LinearSVC(), X_train_tfidf, y_train, X_test_tfidf, y_test)
print("LR  for L1, Count Vectors: ", accuracy_L1)



# Naive Bayes on Word Level TF IDF Vectors
accuracy_L1 = train_model(svm.LinearSVC(), xtrain_count, y_train, xtest_count, y_test)
print("LR  for L1, WordLevel TF-IDF: ", accuracy_L1)



# Naive Bayes on Ngram Level TF IDF Vectors
accuracy_L1 = train_model(svm.LinearSVC(), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram, y_test)
print("LR  for L1, N-Gram Vectors: ", accuracy_L1)



# Naive Bayes on Character Level TF IDF Vectors
accuracy_L1 = train_model(svm.LinearSVC(), xtrain_tfidf_ngram_chars, y_train, xtest_tfidf_ngram_chars, y_test)
print("LR for L1, CharLevel Vectors: ", accuracy_L1)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Part 2: Tokenization

# - **What:** Separate text into units such as sentences or words
# - **Why:** Gives structure to previously unstructured text
# - **Notes:** Relatively easy with English language text, not easy with some languages

# From the [scikit-learn documentation](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):
# 
# > In this scheme, features and samples are defined as follows:
# 
# > - Each individual token occurrence frequency (normalized or not) is treated as a **feature**.
# > - The vector of all the token frequencies for a given document is considered a multivariate **sample**.
# 
# > A **corpus of documents** can thus be represented by a matrix with **one row per document** and **one column per token** (e.g. word) occurring in the corpus.
# 
# > We call **vectorization** the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the **Bag of Words** or "Bag of n-grams" representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.

# In[ ]:


# use CountVectorizer to create document-term matrices from X_train and X_test
vect = CountVectorizer()


# In[ ]:


X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)


# In[ ]:


X_test_dtm.shape


# In[ ]:


# rows are documents, columns are terms (aka "tokens" or "features")
X_train_dtm.shape


# In[ ]:


# last 50 features
print vect.get_feature_names()[-50:]


# In[ ]:


# show vectorizer options
vect


# [CountVectorizer documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

# - **lowercase:** boolean, True by default
# - Convert all characters to lowercase before tokenizing.

# In[ ]:


# don't convert to lowercase
vect = CountVectorizer(lowercase=False)
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape


# - **ngram_range:** tuple (min_n, max_n)
# - The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.

# In[ ]:


get_ipython().run_line_magic('pinfo', 'CountVectorizer')


# In[ ]:


# include 1-grams and 2-grams
vect = CountVectorizer(ngram_range=(1, 2))
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape


# In[ ]:


# last 50 features
print vect.get_feature_names()[-50:]


# In[58]:


#Calculate tf-idf:
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=1)
tfidf = vect.fit_transform(["New Year's Eve in New York",
                            "New Year's Eve in London",
                            "York is closer to London than to New York",
                            "London is closer to Bucharest than to New York"])

#Calculate cosine similarity:
cosine=(tfidf * tfidf.T).A
print(cosine)


# **Predicting the star rating:**

# In[ ]:


# use Naive Bayes to predict the star rating
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)

# calculate accuracy
print metrics.accuracy_score(y_test, y_pred_class)


# In[ ]:


# calculate null accuracy
y_test_binary = np.where(y_test==5, 1, 0)
max(y_test_binary.mean(), 1 - y_test_binary.mean())


# In[ ]:


# define a function that accepts a vectorizer and calculates the accuracy
def tokenize_test(vect):
    X_train_dtm = vect.fit_transform(X_train)
    print 'Features: ', X_train_dtm.shape[1]
    X_test_dtm = vect.transform(X_test)
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    print 'Accuracy: ', metrics.accuracy_score(y_test, y_pred_class)


# In[ ]:


# include 1-grams and 2-grams
vect = CountVectorizer(ngram_range=(1, 2))
tokenize_test(vect)


# ## Part 3: Stopword Removal

# - **What:** Remove common words that will likely appear in any text
# - **Why:** They don't tell you much about your text

# In[ ]:


# show vectorizer options
vect


# - **stop_words:** string {'english'}, list, or None (default)
# - If 'english', a built-in stop word list for English is used.
# - If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
# - If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop words based on intra corpus document frequency of terms.

# In[ ]:


# remove English stop words
vect = CountVectorizer(stop_words='english')
tokenize_test(vect)


# In[ ]:


# set of stop words
print vect.get_stop_words()


# ## Part 4: Other CountVectorizer Options

# - **max_features:** int or None, default=None
# - If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.

# In[ ]:


# remove English stop words and only keep 100 features
vect = CountVectorizer(stop_words='english', max_features=100)
tokenize_test(vect)


# In[ ]:


# all 100 features
print vect.get_feature_names()


# In[ ]:


# include 1-grams and 2-grams, and limit the number of features
vect = CountVectorizer(ngram_range=(1, 2), max_features=100000)
tokenize_test(vect)


# - **min_df:** float in range [0.0, 1.0] or int, default=1
# - When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts.

# In[ ]:


# include 1-grams and 2-grams, and only include terms that appear at least 2 times
vect = CountVectorizer(ngram_range=(1, 2), min_df=2)
tokenize_test(vect)


# ## Part 5: Introduction to TextBlob

# TextBlob: "Simplified Text Processing"

# In[ ]:


print(yelp.text[0])


# In[ ]:


# print the first review
print(yelp.text[0])


# In[ ]:


# save it as a TextBlob object
review = TextBlob(yelp.text[0])


# In[ ]:


print(dir(review))


# In[ ]:


print(review.ngrams(2))


# In[ ]:


review.sentiment


# In[ ]:


# list the words
review.words


# In[ ]:


# list the sentences
review.sentences


# In[ ]:


# some string methods are available
review.lower()


# In[ ]:


review.ngrams(n=2)


# ## Part 6: Stemming and Lemmatization

# **Stemming:**
# 
# - **What:** Reduce a word to its base/stem/root form
# - **Why:** Often makes sense to treat related words the same way
# - **Notes:**
#     - Uses a "simple" and fast rule-based approach
#     - Stemmed words are usually not shown to users (used for analysis/indexing)
#     - Some search engines treat words with the same stem as synonyms

# In[ ]:


# initialize stemmer
stemmer = SnowballStemmer('english')
stemmer


# In[ ]:


review.words


# In[ ]:


# stem each word
print [stemmer.stem(word) for word in review.words]


# **Lemmatization**
# 
# - **What:** Derive the canonical form ('lemma') of a word
# - **Why:** Can be better than stemming
# - **Notes:** Uses a dictionary-based approach (slower than stemming)

# In[ ]:


review.words


# In[ ]:


# assume every word is a noun
print [word.lemmatize() for word in review.words]


# In[ ]:


# assume every word is a verb
print [word.lemmatize(pos='v') for word in review.words]


# In[ ]:


# define a function that accepts text and returns a list of lemmas
def split_into_lemmas(text):
    text = unicode(text, 'utf-8').lower()
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]


# In[ ]:


# use split_into_lemmas as the feature extraction function (WARNING: SLOW!)
vect = CountVectorizer(analyzer=split_into_lemmas)
tokenize_test(vect)


# In[ ]:


# last 50 features
print vect.get_feature_names()[-50:]


# ## Part 7: Term Frequency-Inverse Document Frequency (TF-IDF)

# - **What:** Computes "relative frequency" that a word appears in a document compared to its frequency across all documents
# - **Why:** More useful than "term frequency" for identifying "important" words in each document (high frequency in that document, low frequency in other documents)
# - **Notes:** Used for search engine scoring, text summarization, document clustering

# In[ ]:


# example documents
simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']


# In[ ]:


# Term Frequency
vect = CountVectorizer()
tf = pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())
tf


# In[ ]:


# Document Frequency
vect = CountVectorizer(binary=True)
df = vect.fit_transform(simple_train).toarray().sum(axis=0)
pd.DataFrame(df.reshape(1, 6), columns=vect.get_feature_names())


# In[ ]:


# Term Frequency-Inverse Document Frequency (simple version)
tf/df


# In[ ]:


# TfidfVectorizer
vect = TfidfVectorizer()
pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())


# **More details:** [TF-IDF is about what matters](http://planspace.org/20150524-tfidf_is_about_what_matters/)

# ## Part 8: Using TF-IDF to Summarize a Yelp Review
# 
# Reddit's autotldr uses the [SMMRY](http://smmry.com/about) algorithm, which is based on TF-IDF!

# In[ ]:


get_ipython().run_line_magic('pinfo', 'TfidfVectorizer')


# In[ ]:


# create a document-term matrix using TF-IDF
vect = TfidfVectorizer(stop_words='english')
dtm = vect.fit_transform(yelp.text)
features = vect.get_feature_names()
dtm.shape


# In[ ]:


def summarize():
    
    # choose a random review that is at least 300 characters
    review_length = 0
    while review_length < 300:
        review_id = np.random.randint(0, len(yelp))
        review_text = unicode(yelp.text[review_id], 'utf-8')
        review_length = len(review_text)
    
    # create a dictionary of words and their TF-IDF scores
    word_scores = {}
    for word in TextBlob(review_text).words:
        word = word.lower()
        if word in features:
            word_scores[word] = dtm[review_id, features.index(word)]
    
    # print words with the top 5 TF-IDF scores
    print 'TOP SCORING WORDS:'
    top_scores = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for word, score in top_scores:
        print word
    
    # print 5 random words
    print '\n' + 'RANDOM WORDS:'
    random_words = np.random.choice(word_scores.keys(), size=5, replace=False)
    for word in random_words:
        print word
    
    # print the review
    print '\n' + review_text


# In[ ]:


summarize()


# ## Part 9: Sentiment Analysis

# In[ ]:


print review


# In[ ]:


# polarity ranges from -1 (most negative) to 1 (most positive)
review.sentiment.polarity


# In[ ]:


# understanding the apply method
yelp['length'] = yelp.text.apply(len)
yelp.head(1)


# In[ ]:


def detect_sentiment(text):
    return TextBlob(text.decode('utf-8')).sentiment.polarity


# In[ ]:


# create a new DataFrame column for sentiment (WARNING: SLOW!)
yelp['sentiment'] = yelp.text.apply(detect_sentiment)


# In[ ]:


yelp.head(5)


# In[ ]:


# box plot of sentiment grouped by stars
yelp.boxplot(column='sentiment', by='stars')


# In[ ]:


# reviews with most positive sentiment
yelp[yelp.sentiment == 1].text.head()


# In[ ]:


# reviews with most negative sentiment
yelp[yelp.sentiment == -1].text.head()


# In[ ]:


# widen the column display
pd.set_option('max_colwidth', 500)


# In[ ]:


# negative sentiment in a 5-star review
print yelp[(yelp.stars == 5) & (yelp.sentiment < -0.3)].text


# In[ ]:


# positive sentiment in a 1-star review
print yelp[(yelp.stars == 1) & (yelp.sentiment > 0.5)].text 


# In[ ]:


# reset the column display width
pd.reset_option('max_colwidth')


# ### Adding Features to a Document-Term Matrix

# In[86]:


# create a DataFrame that only contains the 5-star and 1-star reviews
#yelp_best_worst = yelp[(yelp.stars==5) | (yelp.stars==1)]

# define X and y
feature_cols = ['text', 'sentiment', 'cool', 'useful', 'funny']
X = yelp[feature_cols]
y = yelp.stars

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[88]:


# use CountVectorizer with text column only
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train.text)
X_test_dtm = vect.transform(X_test.text)
print(X_train_dtm.shape)
print(X_test_dtm.shape)


# In[89]:


# shape of other four feature columns
X_train.drop('text', axis=1).shape


# In[94]:


# cast other feature columns to float and convert to a sparse matrix
extra = sparse.csr_matrix(X_train.drop('text', axis=1).astype(float))
extra.shape


# In[95]:


# combine sparse matrices
X_train_dtm_extra = sparse.hstack((X_train_dtm, extra))
X_train_dtm_extra.shape


# In[96]:


# repeat for testing set
extra = sparse.csr_matrix(X_test.drop('text', axis=1).astype(float))
X_test_dtm_extra = sparse.hstack((X_test_dtm, extra))
X_test_dtm_extra.shape


# In[98]:


# use logistic regression with text column only
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train_dtm, y_train)
y_pred_class = logreg.predict(X_test_dtm)
print(metrics.accuracy_score(y_test, y_pred_class))


# In[99]:


# use logistic regression with all features
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train_dtm_extra, y_train)
y_pred_class = logreg.predict(X_test_dtm_extra)
print(metrics.accuracy_score(y_test, y_pred_class))


# ## Bonus: Fun TextBlob Features

# In[100]:


# spelling correction
TextBlob('15 minuets late').correct()


# In[103]:


s="this is bcz"


# In[104]:


TextBlob(s).correct()


# In[107]:


# spellcheck
Word('parot').spellcheck()


# In[ ]:


# definitions
Word('bank').define('v')


# In[ ]:


# language identification
TextBlob('Hola amigos').detect_language()


# ## Conclusion
# 
# - NLP is a gigantic field
# - Understanding the basics broadens the types of data you can work with
# - Simple techniques go a long way
# - Use scikit-learn for NLP whenever possible
