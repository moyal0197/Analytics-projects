#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS

import string
import re   
import nltk
from nltk.corpus import stopwords

from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize

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

from textblob import TextBlob
from nltk.stem import PorterStemmer
import nltk
from textblob import Word

#settings

color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))


lem = WordNetLemmatizer()


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data=pd.read_excel('C:\\Users\\ADMIN\\Desktop\\Python ASsignments\\Questions\\Advance\\TEXT\\Bank Reviews\\BankReviews.xlsx')


# In[6]:


data.head()


# In[7]:


data.groupby(['BankName','Stars']).count()


# In[8]:


df = data


# In[9]:


df['text'] = df['Reviews'].astype(str)
df['count_sent']=df["Reviews"].apply(lambda x: len(re.findall("\n",str(x)))+1)

#Word count in each comment:
df['count_word']=df["Reviews"].apply(lambda x: len(str(x).split()))

#Unique word count
df['count_unique_word']=df["Reviews"].apply(lambda x: len(set(str(x).split())))

#Letter count
df['count_letters']=df["Reviews"].apply(lambda x: len(str(x)))

#Word density

df['word_density'] = df['count_letters'] / (df['count_word']+1)

#punctuation count
df["count_punctuations"] =df["Reviews"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))


# In[10]:


df


# In[11]:


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

df['noun_count'] = df['Reviews'].apply(lambda x: check_pos_tag(x, 'noun'))
df['verb_count'] = df['Reviews'].apply(lambda x: check_pos_tag(x, 'verb'))
df['adj_count']  = df['Reviews'].apply(lambda x: check_pos_tag(x, 'adj'))
df['adv_count']  = df['Reviews'].apply(lambda x: check_pos_tag(x, 'adv'))
df['pron_count'] = df['Reviews'].apply(lambda x: check_pos_tag(x, 'pron')) 


# In[12]:


#To check sentiment


# In[13]:


df['sentiment'] = df["Reviews"].apply(lambda x: TextBlob(x).sentiment.polarity )


# In[14]:


#Splitting data into training and Testing,
#Defining data
x=data.Stars
y=data.Reviews
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[15]:


data.shape


# In[16]:


#Data Cleansing


# In[17]:


#Abbrevations and Words correction
def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r' +', ' ', text)
    text = re.sub(r"[-()\"#/@;:{}`+=~|.!?,'0-9]", "", text)
    return(text)


# In[18]:


stop = set(nltk.corpus.stopwords.words('english'))


# In[19]:


stemmer_func = nltk.stem.snowball.SnowballStemmer("english").stem


# In[20]:


import string
def pre_process(text):
  #  text = text.str.replace('/','')
   # text = text.apply(lambda x: re.sub("  "," ", x))
   # text = re.sub(r"[-()\"#/@;:{}`+=~|.!?,']", "", text)
   # text = re.sub(r'[0-9]+', '', text)
    #text = text.apply(lambda x: " ".join(x.translate(str.maketrans('', '', string.punctuation)) for x in x.split() if x.isalpha()))
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    text = text.apply(lambda x: str(TextBlob(x).correct()))
    #text = text.apply(lambda x: " ".join(PorterStemmer().stem(word) for word in x.split()))
    #text = text.apply(lambda x: " ".join(stemmer_func(word) for word in x.split()))
    #text = text.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    #text = text.apply(lambda x: " ".join(word for word, pos in pos_tag(x.split()) if pos not in ['NN','NNS','NNP','NNPS']))
    return(text)


# In[21]:


y_train = y_train.apply(lambda x: clean_text(x))
y_test =y_test.apply(lambda x: clean_text(x))


# In[ ]:


y_train=pre_process(y_train)
y_test=pre_process(y_test)


# In[22]:


#Vecorization


# In[23]:


get_ipython().run_line_magic('pinfo', 'CountVectorizer')


# In[24]:


#Train
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1 ), min_df=5, encoding='latin-1' , max_features=800)
ytrain_count = count_vect.fit_transform(y_train)


# In[25]:


dtm=ytrain_count.toarray()
dtm


# In[26]:


dtm1=pd.DataFrame(dtm)
dtm1.columns=count_vect.get_feature_names()
dtm1.head()


# In[27]:


#Train
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1 ), min_df=5, encoding='latin-1' , max_features=800)
ytrain_count = count_vect.fit_transform(y_train)

tfidf_transformer = TfidfTransformer()
y_train_tfidf = tfidf_transformer.fit_transform(ytrain_count)

#Test
#count_vect = CountVectorizer()
ytest_count = count_vect.transform(y_test)

#tfidf_transformer = TfidfTransformer()
y_test_tfidf = tfidf_transformer.transform(ytest_count)


# In[28]:


dtm2=pd.DataFrame(y_train_tfidf.toarray(), columns=count_vect.get_feature_names())
dtm2.head(100)


# In[29]:


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), max_features=800)
tfidf_vect_ngram.fit(df['Reviews'])
ytrain_tfidf_ngram =  tfidf_vect_ngram.transform(y_train)
ytest_tfidf_ngram =  tfidf_vect_ngram.transform(y_test)


# In[30]:


# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=800)
tfidf_vect_ngram_chars.fit(df['Reviews'])
ytrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(y_train) 
ytest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(y_test)


# In[31]:


#Training LDA model


# In[32]:


lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
y_topics = lda_model.fit_transform(y_train_tfidf)
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()


# In[33]:


# view the topic models
n_top_words = 50
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))

topic_summaries


# In[34]:


frequency_words_wo_stop= {}
for data in df['Reviews']:
    tokens = nltk.wordpunct_tokenize(data.lower())
    for token in tokens:
        if token.lower() not in stop:
            if token in frequency_words_wo_stop:
                count = frequency_words_wo_stop[token]
                count = count + 1
                frequency_words_wo_stop[token] = count
            else:
                frequency_words_wo_stop[token] = 1

            


# In[35]:


frequency_words_wo_stop


# In[36]:


def train_model(classifier, feature_vector_train, label, feature_vector_valid,  valid_x, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_x)


# In[37]:


#Using Naive Bayes


# In[38]:


# Naive Bayes on Count Vectors and TF-IDF
accuracy_L1 = train_model(naive_bayes.MultinomialNB(), y_train_tfidf, x_train, y_test_tfidf, x_test)
print("NB  for L1, Count Vectors: ", accuracy_L1)



# Naive Bayes on Word Level TF IDF Vectors
accuracy_L1 = train_model(naive_bayes.MultinomialNB(), ytrain_count, x_train, ytest_count, x_test)
print("NB  for L1, WordLevel TF-IDF: ", accuracy_L1)



# Naive Bayes on Ngram Level TF IDF Vectors
accuracy_L1 = train_model(naive_bayes.MultinomialNB(), ytrain_tfidf_ngram, x_train, ytest_tfidf_ngram, x_test)
print("NB  for L1, N-Gram Vectors: ", accuracy_L1)



# Naive Bayes on Character Level TF IDF Vectors
accuracy_L1 = train_model(naive_bayes.MultinomialNB(), ytrain_tfidf_ngram_chars, x_train, ytest_tfidf_ngram_chars, x_test)
print("NB for L1, CharLevel Vectors: ", accuracy_L1)


# In[39]:


#Tokenization of data and predicting star rating


# In[40]:


vect = CountVectorizer()


# In[65]:


y_train_dtm = vect.fit_transform(y_train)
y_test_dtm = vect.transform(y_test)


# In[66]:


vect = CountVectorizer(lowercase=False)
y_train_dtm = vect.fit_transform(y_train)
y_train_dtm.shape


# In[67]:


# include 1-grams and 2-grams
vect = CountVectorizer(ngram_range=(1, 2))
y_train_dtm = vect.fit_transform(y_train)
y_train_dtm.shape


# In[68]:


#Predicting Star Rating


# In[69]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(y_train_dtm, x_train)
x_pred_class = nb.predict(y_test_dtm)

# calculate accuracy
print (metrics.accuracy_score(x_test, x_pred_class))


# In[70]:


# define a function that accepts a vectorizer and calculates the accuracy
def tokenize_test(vect):
    y_train_dtm = vect.fit_transform(y_train)
    print ('Features: ', y_train_dtm.shape[1])
    y_test_dtm = vect.transform(y_test)
    nb = MultinomialNB()
    nb.fit(y_train_dtm, x_train)
    y_pred_class = nb.predict(y_test_dtm)
    print ('Accuracy: ', metrics.accuracy_score(x_test, x_pred_class))


# In[71]:


vect = CountVectorizer(ngram_range=(1, 2))
tokenize_test(vect)


# In[72]:


#Sentiment Anlaysis
#To classify reviews into positive neutral and negative


# In[73]:


def detect_sentiment(text):
    return TextBlob(text.decode('utf-8')).sentiment.polarity


# In[74]:


#Although this function can be used to calculate sentiment polarity, it is skipped for now as our df already has polarity that was calculated earlier while exploratory data analysis


# In[75]:


# box plot of sentiment grouped by stars
df.boxplot(column='sentiment', by='Stars')


# In[76]:


# reviews with most positive sentiment
df[df.sentiment == 1].text.head()


# In[77]:


# reviews with most negative sentiment
df[df.sentiment == -1].text.head()


# In[78]:


# reviews with most neutral sentiment
df[df.sentiment == 0].text.head()


# In[79]:


# widen the column display
pd.set_option('max_colwidth', 500)


# In[80]:


df


# In[81]:


# negative sentiment in a 5-star review
print (df[(df.Stars == 5) & (df.sentiment < -0.3)].text)


# In[82]:


# positive sentiment in a 1-star review
print( df[(df.Stars == 1) & (df.sentiment > 0.5)].text )


# In[ ]:




