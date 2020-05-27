from corpus import get_nlf_articles_ads_supplements, sample_articles_per_year, filter_rare_tokens, get_suometar_articles
from gensim import corpora
from gensim.models import LdaSeqModel
import numpy as np
import pickle


# get lemmatized Uusi Suometar articles
start_year = 1854
end_year = 1910
min_article_length = 20
suometar_lemma_filepath = "/wrk/users/zosa/codes/pimlico_store/suometar_tm_train_mine/main/lemmatize/lemmas/data/"
suometar_lemma_filepath_reocr = "/wrk/users/zosa/codes/pimlico_store/suometar_tm_train/main/lemmatize/lemmas/data/"
nlf_filepath = "/wrk/group/newseye/corpora/natlibfin_text_langs/lemmatized_fi.tar"
articles, ads, sup = get_nlf_articles_ads_supplements(start_year=start_year, end_year=end_year, min_article_len=min_article_length)
#articles = get_nlf_articles_by_day_and_page(start_year=start_year, end_year=end_year)
#articles = get_suometar_articles(suometar_lemma_filepath_reocr)

# subsample articles per year so we have flat distribution of articles over time
docs_per_year = 2000
articles_years = sorted(list(articles.keys()))
articles_sorted = [articles[year] for year in articles_years]
articles_sampled, timeslices = sample_articles_per_year(articles_sorted, articles_years, docs_per_year=docs_per_year)
#articles_sampled, timeslices = sample_documents_per_year_by_cluster(articles_sorted, articles_years, docs_per_year=docs_per_year)

# filter our very rare tokens to reduce the vocabulary
min_token_count = 50
articles_filtered = filter_rare_tokens(articles_sampled, min_token_count)
print("Done processing all articles!")

# sanity check
print("Sample article: \n")
random_index = np.random.randint(0, len(articles_filtered))
print([token for token in articles_filtered[random_index]])
print("Length of sample article:", str(len(articles_filtered[random_index])))
print("Total articles:", str(len(articles_filtered)))

# train DTM
print("\nCreating common_corpus and common_dictionary")
common_dictionary = corpora.Dictionary(articles_filtered)
common_corpus = [common_dictionary.doc2bow(doc) for doc in articles_filtered]
print("Size of vocabulary: ", str(len(common_dictionary)))

n_topics = 50
chain_var = 0.08
print("Article years: ", str(articles_years))
print("Time period covered: ", str(len(articles_years)))
print("\nLDASeq Params:")
print("time_slice = ", str(timeslices))
print("n_topics = ", str(n_topics))
print("chain_var = ", str(chain_var))
print("\nStart training Ldaseq model")
ldaseq = LdaSeqModel(corpus=common_corpus,
                     time_slice=timeslices,
                     num_topics=n_topics,
                     id2word=common_dictionary,
                     chain_variance=chain_var)

model_file = "trained_models/ldaseq_nlf_" + str(n_topics) + "topics_" + str(len(articles_years)) + "_years"
ldaseq.save(model_file)
print("***** Done training! Saved trained model as", model_file, "*****")

# save common_dictionary
f = open(model_file + "_dict.pkl", "wb")
pickle.dump(common_dictionary,f)
f.close()
print("Saved common_dictionary!")

#save common_corpus
f = open(model_file + "_corpus.pkl", "wb")
pickle.dump(common_corpus,f)
f.close()
print("Saved common_corpus!")
