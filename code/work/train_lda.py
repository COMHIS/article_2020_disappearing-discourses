from gensim import corpora
from gensim.models import LdaModel, LdaSeqModel
from gensim.models import LdaMulticore
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from corpus import get_suometar_articles, filter_rare_tokens
import os
import string
import pickle
import tarfile
import numpy as np
import re

exclude = set(string.punctuation)
stopwords_file = '../data/stopwords_fi_freq'
stop_list = open(stopwords_file,'rb').read().decode('utf-8').split("\n")
sw = [s.split()[1].lower() for s in stop_list if len(s.split()) > 1]
stopwords_fi = set(stopwords.words('finnish'))
stopwords_fi.update(sw)


def clean_doc(doc):
    doc = " ".join(doc)
    clean_stop = [i for i in doc.lower().split() if i not in stopwords_fi and len(i) > 2]
    clean_digits = " ".join([i for i in clean_stop if re.match(r'^([\s\d]+)$', i) is None])
    clean_punc = ''.join(ch for ch in clean_digits if ch not in exclude)
    return clean_punc


def get_freq_score(articles):
    print("Getting Frequency scores")
    corpus = [" ".join(art) for art in articles]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    scores = np.max(X, axis=0)
    words = vectorizer.get_feature_names()
    freq_dict = {words[i]:scores[i] for i in range(len(words))}
    return freq_dict


def get_tfidf_score(articles):
    print("Getting TF-IDF scores")
    corpus = [" ".join(art) for art in articles]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    scores = np.max(X,axis=0)
    words = vectorizer.get_feature_names()
    tfidf_dict = {words[i]:scores[i] for i in range(len(words))}
    return tfidf_dict


def prune_vocabulary(articles, vocab_len=5000):
    scores = get_tfidf_score(articles)
    valid_words = [w for _,w in sorted(zip(scores.values(), scores.keys()), reverse=True)]
    print("Original vocab: ", len(valid_words))
    valid_words = valid_words[:vocab_len]
    articles_pruned = []
    print("Pruning articles")
    for art in articles:
        pruned_art = [a for a in art if a in valid_words]
        articles_pruned.append(pruned_art)
    return articles_pruned


def subsample_articles(articles, num_articles=100):
    doc_indexes = list(np.random.choice(range(0, len(articles)), num_articles, replace=False))
    subsampled_articles = [articles[i] for i in doc_indexes]
    return subsampled_articles


def train_lda(articles, n_topics, model_filename):
    common_dictionary = corpora.Dictionary(articles)
    common_corpus = [common_dictionary.doc2bow(d) for d in articles]
    print("Documents: ", str(len(articles)))
    print("Topics: ", n_topics)
    print("Vocab:", len(common_dictionary))
    print("Training LDA...")
    lda = LdaModel(common_corpus, id2word=common_dictionary, num_topics=n_topics, passes=50)
    lda.save(model_filename)
    dict_filename = model_filename + "_dict.pkl"
    pickle.dump(common_dictionary, open(dict_filename, "wb"))
    dict_filename = model_filename + "_corpus.pkl"
    pickle.dump(common_corpus, open(dict_filename, "wb"))
    print("Saved trained LDA model as", model_filename, "!")


