import streamlit as st
st.title('SENTIMENT ANALYSER')

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/Python-ML/Datasets/data.csv')

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopword_list = stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()

def remove_stopword(text):
  tokens = tokenizer.tokenize(text)
  tokens = [token.strip() for token in tokens]
  filtered_tokens = [token for token in tokens if token not in stopword_list]
  filtered_text = ' '.join(filtered_tokens)
  return filtered_text

import contractions
def con(text):
  expand = contractions.fix(text)
  return expand

import re
def remove_sp(text):
  pattern = r'[^A-Za-z0-9\s]'
  text = re.sub(pattern,'',text)
  return text

df['review'] = df['review'].apply(lambda x:x.lower())
df['review'] = df['review'].apply(remove_stopword)
df['review'] = df['review'].apply(con)
df['review'] = df['review'].apply(remove_sp) 

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vs = SentimentIntensityAnalyzer()
df['compound'] = df['review'].apply(lambda x: vs.polarity_scores(x)['compound'])

def senti_an(a):
  senti=0
  if a<=-0.05:
    senti=1
  elif a>=0.05:
    senti=3
  else:
    senti=2
  return(senti)

df['rating'] = df['compound'].apply(lambda x: senti_an(x))

x = df['review']
y = df['rating']

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',SVC(C=10,kernel='rbf',gamma=0.1))])
text_model.fit(x,y)

select = st.text_input('Enter your message:')
select = remove_stopword(select)
select = remove_sp(select)
select = con(select)
select = select.lower()

output = text_model.predict([select])
if output==1:
  st.title("NEGATIVE")
elif output==2:
  st.title("NEUTRAL")
else:
  st.title("POSITIVE")
