from gensim import corpora
from gensim.models import LdaSeqModel
import xml.etree.ElementTree as ET
from gensim.models import LdaModel
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk import word_tokenize
from collections import Counter
from sklearn.cluster import AffinityPropagation
from datetime import timedelta
from datetime import datetime
import numpy as np
import os
import string
import pickle
import tarfile
import re
import pandas as pd

exclude = set(string.punctuation)
stopwords_fi = set(stopwords.words('finnish'))
# stopwords_file = '../data/stopwords_fi_freq'
# stop_list = open(stopwords_file, 'rb').read().decode('utf-8').split("\n")
# sw = [s.split()[1].lower() for s in stop_list if len(s.split()) > 1]
# stopwords_fi.update(sw)
#print("Finnish stopwords:", stopwords_fi)

#natlib_fin_tarfile = "../data/lemmatized_fi.tar"
natlib_fin_tarfile = "/wrk/group/newseye/corpora/natlibfin_text_langs/lemmatized_fi.tar"
tar = tarfile.open(natlib_fin_tarfile, "r")
print("Getting lemmatized NLF data from: ", natlib_fin_tarfile)


def clean_document(doc):
    clean_punc = ''.join(ch if ch not in exclude else '' for ch in doc)
    clean_punc_tokens = word_tokenize(clean_punc)
    clean_stop = [tok for tok in clean_punc_tokens if tok not in stopwords_fi and len(tok) > 3]
    clean_digits = [tok for tok in clean_stop if re.match(r'^([\s\d]+)$', tok) is None]
    return clean_digits


def filter_rare_tokens(all_docs, min_count=100):
    print("Filtering tokens appearing less than", min_count, "times")
    all_tokens = [token for doc in all_docs for token in doc]
    counts = Counter(all_tokens)
    valid_tokens = [token for token in counts.keys() if counts[token] >= min_count]
    filtered_docs = []
    for i in range(len(all_docs)):
        doc = all_docs[i]
        filtered = [token for token in doc if token in valid_tokens and len(token) > 3]
        filtered_docs.append(filtered)
    return filtered_docs


def sample_articles_per_year(documents_sorted, years, docs_per_year=100):
    documents_sampled = []
    for y, year in enumerate(years):
        if len(documents_sorted[y]) > docs_per_year:
            random_indexes = list(np.random.choice(range(0, len(documents_sorted[y])), docs_per_year, replace=False))
            documents_year = [documents_sorted[y][i] for i in random_indexes]
        else:
            documents_year = documents_sorted[y]
        documents_sampled.append(documents_year)
    new_timeslice = [len(documents_sampled[y]) for y in range(len(years))]
    documents_sampled_flat = [doc for documents_year in documents_sampled for doc in documents_year]
    return documents_sampled_flat, new_timeslice


def get_suometar_articles(filepath, min_art_len=100, max_art_len=4000):
    suometar_bow = {}
    tar_files = os.listdir(filepath)
    print("Reading lemmatized articles from:", filepath)
    for tar_file in tar_files:
        tar = tarfile.open(filepath + tar_file, "r")
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                filename = member.name
                #print("Filename: ", filename)
                filename = filename.split("_")
                year = filename[1].split("-")[0]
                if year not in suometar_bow.keys():
                    #print("Add year: ", str(year))
                    suometar_bow[year] = []
                article = f.read().decode("utf-8").lower()
                article_clean = clean_document(article)
                if len(article_clean) > min_art_len and len(article_clean) < max_art_len:
                    suometar_bow[year].append(article_clean)
    # sanity check
    print("Unsorted years: ", str(suometar_bow.keys()))
    suometar_years = sorted(suometar_bow.keys())
    print("\nSorted years: ", str(suometar_years))
    return suometar_bow


def get_nlf_articles(lang='fi', start_year=1820, min_article_len=20):
    articles = {}
    tar = tarfile.open(natlib_fin_tarfile, "r")
    #print("Getting lemmatized NLF data from: ", natlib_fin_tarfile)
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f is not None:
            filename = member.name
            #print("Filename: ", filename)
            file_lang = filename.split("/")[1]
            year = int(filename.split("/")[2])
            if year >= start_year:
                article = f.read().decode("utf-8").lower()
                article_clean = clean_document(article)
                if year not in articles:
                    articles[year] = []
                if len(article_clean) > min_article_len:
                    articles[year].append(article_clean)
    # sanity check
    print("Unsorted years: ", str(articles.keys()))
    years = sorted(articles.keys())
    print("\nSorted years: ", str(years))
    return articles


def get_nlf_articles_ads_supplements(lang='fi', start_year=1854, end_year=1920, min_article_len=20):
    articles = {}
    advertisements = {}
    supplements = {}
    tar = tarfile.open(natlib_fin_tarfile, "r")
    print("Getting lemmatized NLF data from: ", natlib_fin_tarfile)
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f is not None:
            filename = member.name
            print("Filename: ", filename)
            year = int(filename.split("/")[2])
            if start_year <= year <= end_year and "swe" not in filename:
                article = f.read().decode("utf-8").lower()
                article_clean = clean_document(article)
                # if "advertisement" in filename:
                #     if year not in advertisements:
                #         advertisements[year] = []
                #     advertisements[year].append(article_clean)
                # elif "supplement" in filename:
                #     if year not in supplements:
                #         supplements[year] = []
                #     supplements[year].append(article_clean)
                if "advertisement" not in filename and "supplement" not in filename:
                    if year not in articles:
                        articles[year] = []
                    if len(article_clean) > min_article_len:
                        articles[year].append(article_clean)
    return articles, advertisements, supplements


def get_nlf_articles_for_year(lang='fi', min_article_len=20, max_articles=10, year=None):
    articles = []
    filenames = []
    members = tar.getmembers()
    for member in members:
        f = tar.extractfile(member)
        if f is not None:
            filename = member.name
            #print("Filename: ", filename)
            file_year = int(filename.split("/")[2])
            if file_year == year and "swe" not in filename and "rus" not in filename:
                article = f.read().decode("utf-8").lower()
                article_clean = clean_document(article)
                if "advertisement" not in filename and "supplement" not in filename:
                    if len(article_clean) > min_article_len:
                        articles.append(article_clean)
                        filenames.append(filename)
        # if len(articles) >= max_articles:
        #     break
    return articles


def get_nlf_metadata(lang='fi', start_year=1900, end_year=1917):
    articles = {'year': [], 'month': [], 'day': [], 'date': [], 'issn': [], 'article_num': []}
    natlib_fin_tarfile = "/wrk/group/newseye/corpora/natlibfin_text_langs/lemmatized_"+lang+".tar"
    tar = tarfile.open(natlib_fin_tarfile, "r")
    print("Getting lemmatized NLF data from: ", natlib_fin_tarfile)
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f is not None:
            filename = member.name
            #print("Filename: ", filename)
            year = int(filename.split("/")[2])
            month_day = filename.split("/")[4].split("_")[1]
            month = int(month_day.split("-")[0])
            day = int(month_day.split("-")[1])
            article_date_str = str(month) + "/" + str(day) + "/" + str(year)
            article_date = datetime.strptime(article_date_str, "%m/%d/%Y")
            if start_year <= year <= end_year and "swe" not in filename and "advertisement" not in filename and "supplement" not in filename:
                issn = filename.split("/")[3]
                article_num = filename.split("/")[-1].split("_")[-1].split(".")[0]
                #print("Article num:", article_num)
                articles['date'].append(article_date)
                articles['year'].append(year)
                articles['month'].append(month)
                articles['day'].append(day)
                articles['issn'].append(issn)
                articles['article_num'].append(article_num)
    df = pd.DataFrame.from_dict(articles)
    unique_years = list(set(df.year))
    for year in unique_years:
        print("Year:", year)
        dfy = df.loc[df.year == year]
        unique_months = list(set(dfy.month))
        for month in unique_months:
            print("Month:", month)
            dfm = dfy.loc[dfy.month == month]
            first_day_month = min(dfm.day)
            first_day_date = dfm.loc[dfm['day'] == first_day_month].iloc[0].date
            day_of_week = first_day_date.weekday()
            if day_of_week >= 5:
                first_day_date += timedelta(days=int(7 % day_of_week))
            week_count = 1
            while first_day_date.month == month:
                print("Week:", week_count)
                dfd = dfm.loc[dfm.date == first_day_date]
                unique_issns = list(set(dfd.issn))
                print("ISSN count:", len(unique_issns))
                for issn in unique_issns:
                    print("ISSN:", issn)
                    dfi = dfd.loc[dfd.issn == issn]
                    art_count = len(list(dfi.article_num))
                    print("Article count:", art_count)
                first_day_date += timedelta(days=7)
                week_count += 1
    return df


def get_nlf_page_metadata(lang='fi', start_year=1820, end_year=1917):
    articles = {'year': [], 'month': [], 'day': [], 'date': [], 'issn': [], 'article_num': [], 'page':[]}
    natlib_fin_tarfile = "/wrk/group/newseye/corpora/natlibfin_text/divisions.tar"
    tar = tarfile.open(natlib_fin_tarfile, "r")
    print("Getting lemmatized NLF data from: ", natlib_fin_tarfile)
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f is not None:
            filename = member.name
            print("Filename:", filename)
            if "swe" not in filename and "rus" not in filename and "metadata" in filename \
                    and "advertisement" not in filename and "supplement" not in filename and "other" not in filename:
                year = int(filename.split("/")[2])
                month_day = filename.split("/")[4].split("_")[1]
                month = int(month_day.split("-")[0])
                day = int(month_day.split("-")[1])
                article_date_str = str(month) + "/" + str(day) + "/" + str(year)
                article_date = datetime.strptime(article_date_str, "%m/%d/%Y")
                #if start_year <= year <= end_year:
                issn = filename.split("/")[3]
                article_num = int(filename.split("/")[-1].split("_")[-1].split(".")[0])
                articles['date'].append(article_date)
                articles['year'].append(year)
                articles['month'].append(month)
                articles['day'].append(day)
                articles['issn'].append(issn)
                articles['article_num'].append(article_num)
                tree = ET.parse(filename)
                root = tree.getroot()
                for ch in root:
                    if ch.tag == 'pages':
                        page_num = int(ch.text.split(",")[0])
                        articles['page'].append(page_num)
        df = pd.DataFrame.from_dict(articles)
    return df


def get_nlf_articles_sampled_weekly(lang='fi', start_year=1900, end_year=1900, min_article_len=50):
    articles = {'year':[], 'month':[], 'day':[], 'date':[], 'text':[], 'issn':[], 'article_num':[]}
    natlib_fin_tarfile = "/wrk/group/newseye/corpora/natlibfin_text_langs/lemmatized_"+lang+".tar"
    tar = tarfile.open(natlib_fin_tarfile, "r")
    print("Getting lemmatized NLF data from: ", natlib_fin_tarfile)
    for member in tar.getmembers():
        f = tar.extractfile(member)
        #if len(articles['year']) > 10000:
        #    break
        if f is not None:
            filename = member.name
            #print("Filename: ", filename)
            year = int(filename.split("/")[2])
            month_day = filename.split("/")[4].split("_")[1]
            month = int(month_day.split("-")[0])
            day = int(month_day.split("-")[1])
            article_date_str = str(month) + "/" + str(day) + "/" + str(year)
            article_date = datetime.strptime(article_date_str, "%m/%d/%Y")
            if start_year <= year <= end_year and "swe" not in filename:
                article = f.read().decode("utf-8").lower()
                article_clean = clean_document(article)
                if "article" in filename and "supplement" not in filename:
                    if len(article_clean) > min_article_len:
                        issn = filename.split("/")[3]
                        article_num = int(filename.split("/")[-1].split("_")[-1].split(".")[0])
                        articles['date'].append(article_date)
                        articles['year'].append(year)
                        articles['month'].append(month)
                        articles['day'].append(day)
                        articles['issn'].append(issn)
                        articles['text'].append(article_clean)
                        articles['article_num'].append(article_num)
    sampled_articles = {}
    df = pd.DataFrame.from_dict(articles)
    unique_years = list(set(articles['year']))
    for year in unique_years:
        print("Sampling article for year:", year)
        df_year = df.loc[df['year'] == year]
        unique_months = sorted(list(set(df_year['month'])))
        for month in unique_months:
            print("Month:", month)
            df_month = df_year.loc[df_year['month'] == month]
            first_day_month = min(df_month.day)
            first_day_date = df_month.loc[df_month['day'] == first_day_month].iloc[0].date
            day_of_week = first_day_date.weekday()
            # weekday: 0-Mon, 1-Tue, 2-Wed, 3-Thur, 4-Fri, 5-Sat, 6-Sun
            if day_of_week >= 5:
                first_day_date += timedelta(days=int(7 % day_of_week))
            while first_day_date.month == month:
                #print("Date:", first_day_date.strftime("%d/%m/%Y"))
                df_date = df_month.loc[df_month['date'] == first_day_date]
                unique_issns = list(set(df_date['issn']))
                for issn in unique_issns:
                    df_issn = df_date.loc[df_date['issn'] == issn]
                    articles_weekly = list(df_issn.text)
                    if year not in sampled_articles:
                        sampled_articles[year] = []
                    if len(articles_weekly) > 10:
                        random_indexes = list(np.random.choice(range(0, len(articles_weekly)), 10, replace=False))
                        articles_weekly_sampled = [articles_weekly[i] for i in random_indexes]
                        sampled_articles[year].extend(articles_weekly_sampled)
                    else:
                        sampled_articles[year].extend(articles_weekly)
                first_day_date += timedelta(days=7)
                #print("Articles:", articles_weekly)
    print("Finished sampling!")
    for year in sampled_articles:
        print("Year:", year)
        print("Articles:", sampled_articles[year])
    return sampled_articles


def train_lda(articles, n_topics, model_filename):
    common_dictionary = corpora.Dictionary(articles)
    common_corpus = [common_dictionary.doc2bow(d) for d in articles]
    print("Documents: ", str(len(articles)))
    print("Topics: ", n_topics)
    print("Training LDA...")
    lda = LdaModel(common_corpus, id2word=common_dictionary, num_topics=n_topics, passes=1000)
    lda.save(model_filename)
    dict_filename = model_filename + "_dict.pkl"
    pickle.dump(common_dictionary, open(dict_filename, "wb"))
    dict_filename = model_filename + "_corpus.pkl"
    pickle.dump(common_corpus, open(dict_filename, "wb"))
    print("Saved trained LDA model as", model_filename, "!")


def train_doc2vec_embeddings(documents, vec_size=100):
    print("Documents:", len(documents))
    tagged_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
    model = Doc2Vec(tagged_documents, vector_size=vec_size, window=5, min_count=10, workers=10)
    return model


def cluster_documents(doc_vectors):
    clustering = AffinityPropagation().fit(doc_vectors)
    labels = clustering.labels_
    return labels


def sample_documents_per_year_by_cluster(documents_sorted, years, vec_size=100, docs_per_year=100):
    documents_sampled = []
    for y, year in enumerate(years):
        print("Year:", year)
        documents_year = documents_sorted[y]
        doc_vecs = train_doc2vec_embeddings(documents_year, vec_size=vec_size)
        doc_clusters = cluster_documents(doc_vecs)
        sampled_indexes = []
        n_clusters = len(set(doc_clusters))
        print("No. of clusters:", n_clusters)
        cluster_counts = Counter(doc_clusters)
        cluster_counts = [cluster_counts[c] for c in range(n_clusters)]
        cluster_prob = [c/sum(cluster_counts) for c in cluster_counts]
        if docs_per_year < n_clusters:
            docs_per_year = n_clusters
        for i in range(docs_per_year):
            # sample cluster
            cluster_num = list(np.random.multinomial(1, cluster_prob, size=1)[0]).index(1)
            # sample article from cluster
            n_docs_cluster = cluster_counts[cluster_num]
            article_num = np.random.choice(range(0, n_docs_cluster))
            article_index = np.where(doc_clusters == cluster_num)[0][article_num]
            while article_index in sampled_indexes:
                article_num = np.random.choice(range(0, n_docs_cluster))
                article_index = np.where(doc_clusters == cluster_num)[0][article_num]
            sampled_indexes.append(article_index)
        sampled_documents_year = [documents_year[i] for i in sampled_indexes]
        documents_sampled.append(sampled_documents_year)
    new_timeslice = [len(docs) for docs in documents_sampled]
    documents_sampled_flat = [doc for documents_year in documents_sampled for doc in documents_year]
    return documents_sampled_flat, new_timeslice

