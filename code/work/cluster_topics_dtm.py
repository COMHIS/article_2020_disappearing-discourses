from gensim.models import LdaSeqModel
import numpy as np
import pickle
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from collections import Counter
from matplotlib import pyplot as plt


def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def plot_dendrogram(model, year, **kwargs):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(16, 10))
    plt.plot()
    dendrogram(linkage_matrix, **kwargs)
    plt.savefig(model_dir + "dendrograms/" + str(year) + ".png")
    #plt.show()

# Load trained Ldaseq model and corpus
start_year = 1854
model_dir = "trained_models/nlf/dtm/64years/"
model_file = model_dir + "ldaseq100_nlf_50topics_64_years"
common_dictionary = pickle.load(open(model_file+"_dict.pkl", "rb"))
common_corpus = pickle.load(open(model_file+"_corpus.pkl", "rb"))
ldaseq = LdaSeqModel.load(model_file)

print("Pre-processing trained TM at", model_file)
n_topics = ldaseq.num_topics
n_timeslices = len(ldaseq.time_slice)
n_docs = np.sum(ldaseq.time_slice)
print("Topics:", n_topics)
print("Timeslices:", n_timeslices)

# top words for each topic in each time slice
topic_words = []
for t in range(n_timeslices):
    words = [" ".join(ldaseq.dtm_coherence(t)[i][:20]) for i in range(n_topics)]
    topic_words.append(words)

# First get TM stats per time slice
lda_stats = []
for t in range(n_timeslices):
    lda_stats.append(ldaseq.dtm_vis(t, common_corpus))

vocab_len = len(lda_stats[0][4])
vocab = lda_stats[0][4]

# Get the term-topic distribution per time slice
topics = []
for i in range(n_timeslices):
    topics.append(lda_stats[i][1])

# Get the document-topic proportions per time slice
doc_topics = []
for i in range(n_timeslices):
    doc_topics.append(np.matrix(lda_stats[i][0]))


def compute_distance_matrix(topic_matrix):
    dist_mat = np.zeros((topic_matrix.shape[0], topic_matrix.shape[0]), dtype=float)
    for k in range(topic_matrix.shape[0]):
        for j in range(topic_matrix.shape[0]):
            if k == j:
                topic_matrix[k][j] = 0.0
            else:
                topic1 = topic_matrix[k]
                topic2 = topic_matrix[j]
                jsd = compute_jsd(topic1, topic2)
                dist_mat[k][j] = jsd
    return dist_mat


for t in range(n_timeslices):
    year = start_year + t
    print("Hierarchical clustering for", year)
    dist_mat = compute_distance_matrix(topics[t])
    model = AgglomerativeClustering(affinity='precomputed', linkage='average', distance_threshold=0, n_clusters=None)
    model = model.fit(dist_mat)
    plot_dendrogram(model, year, truncate_mode=None)