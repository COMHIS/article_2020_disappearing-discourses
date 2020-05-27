from gensim.models import LdaModel
import numpy as np
import os
import pickle
from scipy.stats import entropy
import pandas as pd
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.express as px
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from utilities import topic_to_word_cloud
from collections import defaultdict


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
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def create_pyldavis_plot(lda, common_dictionary, common_corpus):
    vis_data = pyLDAvis.gensim.prepare(lda, common_corpus, common_dictionary, sort_topics=False)
    model_name = model_file.split("/")[-1]
    out_filename = model_dir + "pyldavis/" + model_name + ".html"
    outfile = open(out_filename, 'w')
    pyLDAvis.save_html(vis_data, fileobj=outfile)
    return out_filename


def plot_topic_share_stacked_bar_plot_plotly(df, filename):
    print("Plotting stacked bar plot with plotly")
    topic_ids = list(df['topic_id'].unique())
    valid_topics = []
    for k in topic_ids:
        if df[df['topic_id']==k]['topic_weight'].sum() > 0.0:
            valid_topics.append(k)
    df = df[df.topic_id.isin(valid_topics)]
    corpus_years = list(df['year'].unique())
    fig = go.Figure()
    for k in valid_topics:
        topic_share = list(df[df['topic_id'] == k].topic_weight)
        top_words = list(df[df['topic_id'] == k].topic_words)
        topic_name = "Topic_"+str(k)
        fig.add_trace(go.Bar(x=corpus_years,
                             y=topic_share,
                             name=top_words[0],
                             hovertext=top_words
                             ))
    fig.update_layout(barmode='stack',
                      xaxis={'categoryorder': 'category ascending'},
                      xaxis_tickangle=-45,
                      plot_bgcolor='#fff')
    plotly.offline.plot(fig, filename=model_dir + 'threshold_doc/bar_plots/' + filename, auto_open=True)
    print("Done saving Plotly plot as", filename, "!")


def compute_topic_popularity_by_year(df):
    corpus_years = list(df['year'].unique())
    for year in corpus_years:
        df_year = df[df['year']==year]
        df_year = df_year.assign(topic_rank=df_year['topic_weight'].rank(ascending=False))
        df_year = df_year.sort_values(by='topic_rank')
        csv_file = model_dir + "threshold_doc/topic_rankings/" + str(year) + ".csv"
        df_year.to_csv(csv_file, sep='\t', encoding='utf-8', index=False)


def compute_topic_share_whole_corpus():
    # compute normalized topic shares per year
    print("Computing normalized topic share per year")
    pickle_dir = "results/lda/"
    topic_shares = []
    pickle_files = sorted(os.listdir(pickle_dir))
    for pf in pickle_files:
        data = pickle.load(open(pickle_dir + pf, 'rb'))
        doc_matrix = np.zeros((len(data), n_topics))
        for i,doc in enumerate(data):
            for tup in doc:
                topic_prop = tup[1]
                if topic_prop < topic_thresh:
                    doc_matrix[i,tup[0]] = 0.0
                else:
                    doc_matrix[i,tup[0]] = topic_thresh
            doc_matrix[i] /= doc_matrix[i].sum()
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
            topic_words_list.append(topic_words[k])
            topic_share_list.append(topic_shares[t][k])
            topic_id_list.append(k+1)
            topic_year_list.append(start_year + t)
    df = {"topic_id": topic_id_list,
          "topic_words": topic_words_list,
          "topic_weight": topic_share_list,
          "year": topic_year_list}
    df = pd.DataFrame.from_dict(df)
    csv_file = "disappearing_lda.csv"
    df.to_csv(csv_file, sep='\t', encoding='utf-8', index=False)
    return df, topic_shares


def plot_topic_share_line_plot(topic_shares):
    n_timeslices = len(topic_shares)
    for k in range(n_topics):
        plt.figure(figsize=(16, 10))
        plt.plot()
        topic_prob = [topic_shares[t][k] for t in range(n_timeslices)]
        plt.plot(range(n_timeslices), topic_prob, marker='o', linestyle='-', linewidth=1, label='Topic_'+str(k+1))
        plt.xticks(range(n_timeslices), labels=[str(start_year+y) for y in range(n_timeslices)], rotation='vertical')
        plt.legend(["Topic "+str(k+1)])
        plt.savefig(model_dir + "threshold_doc/line_plots/Topic_"+str(k+1)+".png")
        plt.close()
        #plt.show()


def create_word_clouds():
    topics = lda.get_topics()
    for k in range(n_topics):
        print("topic = ", k)
        topic_dist = topics[k]
        image = topic_to_word_cloud(common_dictionary, topic_dist)
        image_filename = model_dir + "word_clouds/Topic_" + str(k+1) + ".png"
        image.save(image_filename)


start_year = 1854
model_dir = "trained_models/nlf/lda/"
model_file = model_dir + "lda_nlf_50topics"
lda = LdaModel.load(model_file)
common_dictionary = pickle.load(open(model_file + "_dict.pkl", "rb"))
common_corpus = pickle.load(open(model_file+"_corpus.pkl", "rb"))
n_topics = lda.num_topics
vocab_len = len(common_dictionary)
vocab = list(common_dictionary.keys())

topic_words = [" ".join([w[0] for w in lda.show_topic(k)]) for k in range(n_topics)]

topic_thresh = 0.005
df, topic_shares = compute_topic_share_whole_corpus()
#create_word_clouds()
#plot_topic_share_line_plot(topic_shares)
#compute_topic_popularity_by_year(df)
bar_plot_name = "lda_whole_corpus_topic_words.html"
plot_topic_share_stacked_bar_plot_plotly(df, bar_plot_name)
#
# create_pyldavis_plot(lda, common_dictionary, common_corpus)