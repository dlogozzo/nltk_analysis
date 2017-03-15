import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from __future__ import division
from collections import Counter, defaultdict
import pandas as pd
from pandas import DataFrame

bib = nltk.corpus.gutenberg.words(u'bible-kjv.txt')
bibtxt = nltk.Text(bib)

text = ntlk.word_tokenize(text)

out = dict(nltk.pos_tag(text))


def words(text):
    return re.findall(r'\w+', text.lower())

def P(word, N=sum(WORDS.values())):
    """Probability of a word"""
    return WORDS[word] / N

def correction(word):
    """Most probable spelling correction for word"""
    return max(candidates(word), key=P)
    
def lexical_diversity(text):
    return len(set(text)) / len(text)

def percentage(count, total):
    return 100 * count / total
    
def sevens(text):
    seven_list = defaultdict(list)
    #vocab = set(text)
    vocab = set([word.lower() for word in bibtxt if word.isalpha()])
    seven_list = {word : text.count(word) 
                  for word in vocab 
                  if text.count(word) % 7 == 0}
    return seven_list
    
def check_features(text):
    vocab = set([word.lower() for word in bibtxt if word.isalpha()])
    features = [word
               for word in vocab
               if text.count("seven")]
    return features    

bibtxt.count("seven")

def paragraph_regex(text):
    text = [w for w in text if re.search(r'(\d{1,2}:\d{1,2})',w)]
    return text


def compress(word):
    books = re.findall(regexp, word)
    return ''.join(books)

    
df = pd.DataFrame(seven_list.items(),columns=['Word','Count'])
df.sort_values(['Count'],ascending=False,inplace=True)
df['Divisible by 7'] = df['Count'] / 7

df.to_csv('F:/Desktop/sevens.csv',index=False)
