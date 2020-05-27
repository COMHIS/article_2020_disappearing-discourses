from corpus import get_nlf_articles_ads_supplements, sample_articles_per_year, filter_rare_tokens, get_suometar_articles
from gensim import corpora
from train_lda import train_lda
import numpy as np
import pickle


# get lemmatized Uusi Suometar articles
start_year = 1854
end_year = 1917
min_article_length = 20
articles, ads, sup = get_nlf_articles_ads_supplements(start_year=start_year, end_year=end_year, min_article_len=min_article_length)
#articles = get_nlf_articles_by_day_and_page(start_year=start_year, end_year=end_year)
#articles = get_suometar_articles(suometar_lemma_filepath_reocr)

# subsample articles per year so we have flat distribution of articles over time
docs_per_year = 500
years = sorted(list(articles.keys()))
articles_sorted = [articles[year] for year in years]
articles_sampled, timeslices = sample_articles_per_year(articles_sorted, years, docs_per_year=docs_per_year)
#articles_sampled, timeslices = sample_documents_per_year_by_cluster(articles_sorted, articles_years, docs_per_year=docs_per_year)

# filter our very rare tokens to reduce the vocabulary
min_token_count = 40
articles_filtered = filter_rare_tokens(articles_sampled, min_token_count)
print("Done processing all articles!")

# train one big LDA for all timeslices
n_topics = 50
model_filename = "trained_models/lda_nlf_" + str(n_topics) + "topics_" + str(docs_per_year)
train_lda(articles_filtered, n_topics, model_filename)

# train separate LDA for each year
for i,year in enumerate(years):
    start_index = i*docs_per_year
    end_index = (i+1)*docs_per_year
    articles_year = articles_filtered[start_index:end_index]
    model_filename = "trained_models/lda_nlf_" + str(year) + "_" + str(n_topics) + "topics_" + str(docs_per_year)
    train_lda(articles_year, n_topics, model_filename)

