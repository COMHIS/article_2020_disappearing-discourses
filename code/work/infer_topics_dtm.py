from gensim.models import LdaSeqModel
import numpy as np
import random
from collections import Counter


def compute_theta_current(z, n_topics, alphas, old_topic):
    counts = Counter(z)
    m = [counts[k] for k in range(n_topics)]
    m[old_topic] = m[old_topic] - 1
    theta = m + alphas
    theta_norm = theta/theta.sum()
    return theta_norm


def compute_theta_post(z, n_topics, alphas):
    counts = Counter(z)
    m = [counts[k] for k in range(n_topics)]
    theta = m + alphas
    theta_norm = theta/theta.sum()
    return theta_norm


def get_topic_for_word(word, topics, samples=10):
    n_topics = topics.shape[0]
    phi = topics
    word_id = word
    word_prob = np.array([phi[k, word_id] for k in range(n_topics)])
    word_prob /= word_prob.sum()
    sampled_topics = []
    for i in range(samples):
        sampled_topic = list(np.random.multinomial(1, word_prob, size=1)[0]).index(1)
        sampled_topics.append(sampled_topic)
    counts = Counter(sampled_topics)
    word_topic = [i for _,i in sorted(zip(counts.values(), counts.keys()), reverse=True)][0]
    return word_topic


def get_topic_dist_from_model(model, new_documents, time_slice):
    new_corpus = [model.id2word.doc2bow(doc) for doc in new_documents]
    topics = model.dtm_vis(time_slice, new_corpus)[1]
    return topics, new_corpus


def get_new_doc_topics_old(model, new_documents, time_slice):
    doc_topics = []
    topics, new_corpus = get_topic_dist_from_model(model, new_documents, time_slice)
    alphas = np.tile(model.alphas, (1, 1))
    n_topics = len(topics)
    for doc in new_corpus:
        doc_tokens = [[w[0]]*w[1] for w in doc]
        doc_tokens2 = [w for sub_list in doc_tokens for w in sub_list]
        z = [get_topic_for_word(w, topics) for w in doc_tokens2]
        theta = compute_theta_post(z, n_topics, alphas)
        doc_topics.append(theta)
    return doc_topics


def get_new_doc_topics(model, new_documents, time_slice, n_iter=100):
    alphas = model.alphas
    topics, new_corpus = get_topic_dist_from_model(model, new_documents, time_slice)
    n_topics = len(topics)
    documents = []
    z = []
    for doc in new_corpus:
        tokens_doc = [[w[0]] * w[1] for w in doc]
        tokens_doc = [w for sub_list in tokens_doc for w in sub_list]
        z_doc = [random.randrange(0, n_topics) for _ in range(len(tokens_doc))]
        z.append(z_doc)
        documents.append(tokens_doc)
    for iter in range(n_iter):
        for j, doc in enumerate(documents):
            for i, word_id in enumerate(doc):
                old_topic = z[j][i]
                theta_doc = compute_theta_current(z[j], n_topics, alphas, old_topic)
                phi_word = topics.T[word_id]
                topic_prob = theta_doc * phi_word
                topic_prob = topic_prob / topic_prob.sum()
                new_topic = list(np.random.multinomial(1, topic_prob, size=1)[0]).index(1)
                z[j][i] = new_topic
    new_doc_topics = []
    for i, z_doc in enumerate(z):
        theta_post = compute_theta_post(z_doc, n_topics, alphas)
        new_doc_topics.append(theta_post)
    return new_doc_topics
