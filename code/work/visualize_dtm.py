#################################################################################
# Visualizing Dynamic Topic Models
#################################################################################

from gensim.models import LdaSeqModel
import numpy as np
import os
import pickle
import random
from scipy.stats import entropy
import pandas as pd
import seaborn as sns
from collections import Counter
from utilities import topic_to_word_cloud
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
import plotly
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA


# Load trained Ldaseq model and corpus
start_year = 1854
topic_thresh = 0.01
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
term_topic_stats = []
for i in range(n_timeslices):
    term_topic_stats.append(lda_stats[i][1])

# Get the document-topic proportions per time slice
doc_topics_stats = []
for i in range(n_timeslices):
    doc_topics_stats.append(np.matrix(lda_stats[i][0]))

#def compute_normalized_topic_shares():
# compute normalized topic shares per year
print("Computing normalized topic share per year")
topic_shares = []
num_docs_per_ts = ldaseq.time_slice
end_indexes = np.cumsum(num_docs_per_ts)
doc_topics = doc_topics_stats[0]
for i in range(len(end_indexes)):
    if i == 0:
        start_index = 0
    else:
        start_index = end_indexes[i-1]
    end_index = end_indexes[i]
    topic_share_ts = []
    for k in range(n_topics):
        topic_sum = np.sum(doc_topics[start_index:end_index,k])
        topic_share_ts.append(topic_sum)
    topic_share_ts = np.array(topic_share_ts)
    topic_share_ts /= np.sum(topic_share_ts)
    topic_share_ts[topic_share_ts < topic_thresh] = 0.0
    topic_share_ts /= np.sum(topic_share_ts)
    topic_shares.append(topic_share_ts)

# create Dataframe from dict
topic_words_list = []
topic_id_list = []
topic_year_list = []
topic_share_list = []

for t in range(n_timeslices):
    for k in range(n_topics):
        topic_words_list.append(topic_words[t][k])
        #topic_words_list.append(k+1)
        topic_share_list.append(topic_shares[t][k])
        topic_id_list.append(k)
        topic_year_list.append(start_year+t)

df = {"topic_id": topic_id_list,
      "topic_words": topic_words_list,
      "topic_weight": topic_share_list,
      "year": topic_year_list}
df = pd.DataFrame.from_dict(df)

topic_names = ["Topic_"+str(k+1) for k in range(n_topics)]
topic_ids = [i for i in range(n_topics)]
labels = {"topic_id": topic_ids,
          "topic_labels": topic_names}
labels = pd.DataFrame.from_dict(labels)


def normalise_matrix_rows(mat):
    mat_norm = mat / np.linalg.norm(mat, ord=2, axis=1, keepdims=True)
    return mat_norm


def compute_term_frequency_in_corpus(corpus):
    term_freq = [0 for _ in range(vocab_len)]
    for doc in corpus:
        for word_tuple in doc:
            term_freq[word_tuple[0]] += word_tuple[1]
    return term_freq


def compute_lift(common_corpus, common_dictionary, word_prob):
    vocab_size = len(common_dictionary)
    lift_score = [0 for _ in range(vocab_size)]
    for doc in common_corpus:
        for word_tuple in doc:
            lift_score[word_tuple[0]] += word_tuple[1]
    for i in range(vocab_size):
        lift_score[i] = word_prob[i] / (lift_score[i]*vocab_size)
    return lift_score


def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def compute_topic_shares_whole_corpus():
    # def compute_normalized_topic_shares():
    # compute normalized topic shares per year
    print("Computing normalized topic share per year")
    pickle_dir = "results/dtm/"
    topic_shares = []
    pickle_files = sorted(os.listdir(pickle_dir))
    for pf in pickle_files:
        print(pf)
        data = pickle.load(open(pickle_dir + pf, 'rb'))
        #data = [d[0] for d in data]
        doc_matrix = np.stack(data)
        doc_matrix[doc_matrix < topic_thresh] = 0.0
        doc_matrix = normalise_matrix_rows(doc_matrix)
        topic_share_ts = doc_matrix.sum(axis=0)
        topic_share_ts /= np.sum(topic_share_ts)
        #topic_share_ts[topic_share_ts < topic_thresh] = 0.0
        #topic_share_ts /= np.sum(topic_share_ts)
        topic_shares.append(topic_share_ts)
    # create Dataframe from dict
    topic_words_list = []
    topic_id_list = []
    topic_year_list = []
    topic_share_list = []
    for t in range(len(pickle_files)):
        for k in range(n_topics):
            topic_words_list.append(topic_words[t][k])
            topic_share_list.append(topic_shares[t][k])
            topic_id_list.append(k)
            topic_year_list.append(start_year + t)
    df = {"topic_id": topic_id_list,
          "topic_words": topic_words_list,
          "topic_weight": topic_share_list,
          "year": topic_year_list}
    df = pd.DataFrame.from_dict(df)
    csv_file = "disappearing_dtm.csv"
    df.to_csv(csv_file, sep='\t', encoding='utf-8', index=False)
    return df, topic_shares


def plot_topic_share_line_plot(topic_shares):
    print("Plotting topic share line plots")
    for k in range(n_topics):
        plt.figure(figsize=(16, 10))
        plt.plot()
        topic_prob = [topic_shares[t][k] for t in range(len(topic_shares))]
        plt.plot(range(len(topic_shares)), topic_prob, marker='o', linestyle='-', linewidth=1, label='Topic_'+str(k+1))
        plt.xticks(range(len(topic_shares)), labels=[str(start_year+y) for y in range(len(topic_shares))], rotation='vertical')
        #plt.legend(["Topic "+str(k+1)])
        plt.savefig(model_dir + "threshold_doc/line_plots/Topic_" + str(k+1) + ".png")
        plt.close()
        #plt.show()


def plot_topic_share_stacked_bar_plot_df(df):
    print("Plotting stacked bar plot")
    df2 = df.pivot('year', 'topic_words', 'topic_weight')
    ax = df2.plot(kind='bar',
                  stacked=True,
                  colormap='tab20',
                  edgecolor='black',
                  figsize=(16, 10),
                  title='Topic shares per year')
    topic_names = ["Topic_" + str(n) for n in list(df2.columns)]
    plt.legend(topic_names, bbox_to_anchor=(1.1, 1.05))
    fig = ax.get_figure()
    fig.savefig(model_dir + 'bar_plots/stacked_barplot_topic_share.png')


def plot_topic_share_stacked_bar_plot_plotly(df, filename):
    print("Plotting stacked bar plot with plotly")
    corpus_years = list(range(start_year, start_year+n_timeslices))
    fig = go.Figure()
    for k in range(n_topics):
        topic_share = list(df[df['topic_id'] == k].topic_weight)
        top_words = list(df[df['topic_id'] == k].topic_words)
        topic_name = "Topic_"+str(k+1)
        fig.add_trace(go.Bar(x=corpus_years,
                             y=topic_share,
                             name=topic_name, #top_words[0],
                             hovertext=top_words
                             ))
    fig.update_layout(barmode='stack',
                      xaxis={'categoryorder': 'category ascending'},
                      xaxis_tickangle=-45,
                      plot_bgcolor='#fff')
    plotly.offline.plot(fig, filename= model_dir + 'threshold_doc/bar_plots/' + filename, auto_open=True)
    print("Done saving Plotly plot as", filename, "!")


def create_word_clouds():
    for k in range(n_topics):
        print("topic = ", k)
        for t in range(n_timeslices):
            print("time = ", start_year+t)
            #topic_dist = term_topic_lift[t][k]
            topic_dist = np.array(term_topic_stats[t][k])[0]
            image = topic_to_word_cloud(common_dictionary, topic_dist)
            image_filename = model_dir + "word_clouds/topic_" + str(k+1) + "_time_" + str(start_year+t) + ".png"
            image.save(image_filename)


def pyldavis_for_time_slice(time_slice=0):
    print("Generating pyLDAvis for year", start_year+time_slice)
    # create one pyLDAvis model for the given time_slice
    end_indexes = np.cumsum(ldaseq.time_slice)
    end_index = end_indexes[time_slice]
    if time_slice == 0:
        start_index = 0
    else:
        start_index = end_indexes[time_slice-1]
    data = {'topic_term_dists': term_topic_stats[time_slice],
            'doc_topic_dists': doc_topics[start_index:end_index, ],
            'doc_lengths': [len(d) for d in common_corpus[start_index:end_index]],
            'vocab': vocab,
            'term_frequency': compute_term_frequency_in_corpus(common_corpus[start_index:end_index])}
    print("Convert data to pyLDAvis data")
    vis_data = pyLDAvis.prepare(**data)
    print("Saving visualization to html")
    outfile = open(model_dir + "pyldavis/" + str(start_year+time_slice) + ".html", 'w')
    pyLDAvis.save_html(vis_data, fileobj=outfile)


def build_heatmap_for_topic(topic_num, words_per_year=20, selected_words=None):
    print("Visualize Topic", topic_num+1, "as heatmap")
    top_topic_words = []
    if selected_words is not None:
        top_topic_words = sorted(selected_words)
    else:
        for t in range(n_timeslices):
            top_words = topic_words[t][topic_num].split()[:words_per_year]
            top_topic_words.extend(top_words)
        top_topic_words = sorted(list(set(top_topic_words)))
    heatmap_dict = {}
    for t in range(n_timeslices):
        word_prob_ts = []
        for word in top_topic_words:
            word_index = vocab.index(word)
            word_prob = term_topic_stats[t][topic_num][word_index]
            word_prob_ts.append(word_prob)
        heatmap_dict[str(start_year+t)] = word_prob_ts
    heatmap_dict['words'] = top_topic_words
    heatmap_df = pd.DataFrame.from_dict(heatmap_dict)
    heatmap_df = heatmap_df.set_index('words')
    print("Creating heatmap")
    plt.figure(figsize=(30, 14))
    sns_plot = sns.heatmap(heatmap_df, yticklabels=1, linewidths=0.5, linecolor='black')
    sns_plot.set(xticklabels=[str(start_year+t) for t in range(n_timeslices)])
    sns_plot.tick_params(labelsize=16)
    sns_plot.set_xticklabels(labels=[str(start_year+t) for t in range(n_timeslices)], rotation=45)
    #sns_plot.set(yticklabels=top_topic_words)
    sns_plot.set_yticklabels(labels=top_topic_words, rotation=0)
    sns_plot.figure.savefig(model_dir + "heatmaps/Topic_"+str(topic_num+1)+".png")


def build_lineplots_for_topic_words(topic_num, words_per_year=5, selected_words=None):
    print("Visualize Topic", topic_num+1, "as lineplot")
    top_topic_words = []
    for t in range(n_timeslices):
        top_words = topic_words[t][topic_num].split()[:words_per_year]
        top_topic_words.extend(top_words)
    top_topic_words = sorted(list(set(top_topic_words)))
    if selected_words is not None:
        top_topic_words = selected_words#sorted(list(set(selected_words).intersection(set(top_topic_words))))
    dictionary = {'word':[], 'year':[], 'prob':[]}
    for t in range(n_timeslices):
        for word in top_topic_words:
            word_index = vocab.index(word)
            word_prob = np.array(term_topic_stats[t][topic_num])[0][word_index]
            dictionary['word'].append(word)
            dictionary['prob'].append(word_prob)
            dictionary['year'].append(start_year+t)
    df = pd.DataFrame.from_dict(dictionary)
    plt.figure(figsize=(20, 10))
    sns_plot = sns.lineplot(x='year',
                            y='prob',
                            hue='word',
                            data=df,
                            linestyle='-',
                            marker='o',
                            dashes=False,
                            palette=sns.color_palette("husl", len(top_topic_words)))
    #sns_plot.set(xticklabels=[str(start_year + t) for t in range(n_timeslices)])
    #sns.FacetGrid.set(xticks=[str(start_year + t) for t in range(n_timeslices)])
    sns_plot.figure.savefig(model_dir + "line_plots_topic_words/Topic_"+str(topic_num+1)+".png")


def print_topic_words(topic_num=0, n_words=10):
    print("----------")
    for t in range(n_timeslices):
        print("Time:", start_year+t)
        words = topic_words[t][topic_num].split()[:n_words]
        for w in words:
            print(w)
        print("----------")


def get_mean_top_topic_words(topic_num=0, n_words=10):
    topic_dists = []
    for t in range(n_timeslices):
        topic = term_topic_stats[t][topic_num]
        topic_dists.append(topic)
    topic_dists = np.mean(np.array(topic_dists), axis=0)
    sorted_indexes = [i for _,i in sorted(zip(topic_dists, range(vocab_len)), reverse=True)]
    mean_top_words = list(np.array(vocab)[sorted_indexes[:n_words]])
    return mean_top_words


def get_top_topic_words_for_timeslice(topic_num=0, timeslice=0, n_words=10):
    words = topic_words[timeslice][topic_num].split()[:n_words]
    return words


def plot_num_articles():
    print("Plotting article stats")
    data = open("trained_models/stats_nlf_all.txt").readlines()
    years = []
    num_articles = []
    mean_art_len = []
    for line in data:
        if "Year" in line:
            y = int(line.strip().split(":")[1])
            years.append(y)
        elif "Articles" in line:
            a = int(line.strip().split(":")[1])
            num_articles.append(a)
        elif "Mean art len" in line:
            m = round(float(line.strip().split(":")[1]),2)
            mean_art_len.append(m)
        elif "ads" in line:
            break
    #fig, ax = plt.subplots()
    #for n,label in enumerate(ax.xaxis.get_ticklabels()):
    #    label.set_visible(True)
    plt.figure(figsize=(20, 12))
    plt.bar(years[10:40], num_articles[10:40])
    plt.savefig("trained_models/article_counts_30years.png")
    plt.close()
    plt.figure(figsize=(20, 12))
    plt.bar(years[10:40], mean_art_len[10:40])
    plt.savefig("trained_models/article_length_30years.png")
    plt.close()


def plot_pca_topics_in_timeslice(time_slice=0):
    print("Visualize topics in year", start_year+time_slice)
    df = {}
    df['topic_labels'] = ['Topic_'+str(k+1) for k in range(n_topics)]
    df = pd.DataFrame.from_dict(df)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(term_topic_stats[time_slice])
    df['pca1'] = pca_result[:, 0]
    df['pca2'] = pca_result[:, 1]
    df_subset = df[~df['topic_labels'].isin(['Topic_35', 'Topic_26'])]
    # plotting document clusters
    print("Plotting topics as PCA points")
    ### Plot data with Plotly
    fig = px.scatter(df,
                     x='pca1',
                     y='pca2',
                     color='topic_labels',
                     width=1000,
                     height=700)
    fig.update_traces(mode='markers', marker_size=10)
    #fig.show()
    plotly.offline.plot(fig, filename=model_dir + "/topic_pca/" + str(start_year+time_slice) + ".html", auto_open=False)


def compute_topic_similarity_matrix(cur_time_slice=0, topics_small=10, topic_weights=None):
    next_time_slice = cur_time_slice + 1
    similarity_matrix = np.zeros((topics_small, topics_small), dtype=int)
    for k in range(topics_small):
        for j in range(topics_small):
            if similarity_matrix[k][j] == 0:
                cur_topic = term_topic_stats[cur_time_slice][k]
                next_topic = term_topic_stats[next_time_slice][j]
                jsd = compute_jsd(cur_topic, next_topic)
                if topic_weights is not None:
                    sim = int(((1.0 - jsd) * topic_weights[k])*100)
                else:
                    sim = int((1.0 - jsd) * 100)
                similarity_matrix[k][j] = sim
                similarity_matrix[j][k] = sim
    return similarity_matrix


def plot_sankey_topic_distance():
    print("Plotting Sankey chart of topic distance")
    timeslices = 5
    topics_small = 50
    labels = [["Topic"+str(k+1)+"_"+str(start_year+t) for k in range(topics_small)] for t in range(timeslices)]
    labels = [l for sublist in labels for l in sublist]
    color_dict = {"Topic"+str(k+1):random_color()[0] for k in range(topics_small)}
    source = []
    target = []
    value = []
    min_sim_thresh = 0
    df, topic_shares = compute_topic_shares_whole_corpus()
    for t in range(timeslices-1):
        similarity_matrix = compute_topic_similarity_matrix(t, topics_small=topics_small, topic_weights = topic_shares[t])
        for k in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                val = int((similarity_matrix[k][j] / np.sum(similarity_matrix[k]))*100)
                #val_weighted = round((val * topic_shares[t][k]) * 100)
                #print("val:", val)
                #print("topic_share:", topic_shares[t][k])
                print("val:", val)
                if val > min_sim_thresh:
                    source.append(k + (t*topics_small))
                    target.append(j + (t*topics_small) +topics_small)
                    value.append(val)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=[color_dict[label.split("_")[0]] for label in labels]
        ),
        link=dict(
            source=source,  # indices correspond to labels, eg A1, A2, A2, B1, ...
            target=target,
            value=value
        ))])
    fig.update_layout(title_text="Disappearing Discourses", font_size=10)
    #fig.show()
    plotly.offline.plot(fig, filename=model_dir + "sankey.html", auto_open=False)


for k in range(n_topics):
    print("Topic", k+1, ":")
    selected_words = get_mean_top_topic_words(topic_num=k, n_words=15)
    print("Mean top words:", " ".join(selected_words))
#     #build_lineplots_for_topic_words(topic_num=k, selected_words=selected_words)
#     build_heatmap_for_topic(topic_num=k, selected_words=selected_words, words_per_year=15)

#plot_topic_share_stacked_bar_plot_df()
#plot_topic_share_line_plot()
#create_word_clouds()
#plot_topic_share_stacked_bar_plot_plotly()
plot_filename = "stacked_barplot_whole_corpus_topic_num.html"
df, topic_shares = compute_topic_shares_whole_corpus()
#plot_topic_share_stacked_bar_plot_plotly(df, plot_filename)
#plot_topic_share_line_plot(topic_shares)

#for t in range(n_timeslices):
#    plot_pca_topics_in_timeslice(time_slice=t)
#plot_sankey_topic_distance()