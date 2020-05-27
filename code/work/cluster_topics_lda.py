from gensim.models import LdaModel
import numpy as np
import pickle
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt


def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def plot_dendrogram(model, **kwargs):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    labels = [l+1 for l in model.labels_]
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
    dendrogram(linkage_matrix, labels = labels, **kwargs)
    plt.savefig(model_dir + "dendrograms/" + model_name + "_2.png")
    #plt.show()

# Load trained Ldaseq model and corpus
model_dir = "trained_models/nlf/lda/"
model_name = "lda_nlf_50topics"
model_file = model_dir + model_name
common_dictionary = pickle.load(open(model_file+"_dict.pkl", "rb"))
common_corpus = pickle.load(open(model_file+"_corpus.pkl", "rb"))
lda = LdaModel.load(model_file)




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



print("Hierarchical clustering for LDA model", model_name)
topics = lda.get_topics()
dist_mat = compute_distance_matrix(topics)
model = AgglomerativeClustering(affinity='precomputed', linkage='average', distance_threshold=0, n_clusters=None)
model = model.fit(dist_mat)
plot_dendrogram(model, truncate_mode=None)