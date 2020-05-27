from gensim.models import LdaModel
import pickle
import os
from corpus import get_nlf_articles_for_year


model_dir = "trained_models/lda/"
model_file = model_dir + "lda_nlf_50topics"
print("Opening trained TM at", model_file)

common_dictionary = pickle.load(open(model_file+"_dict.pkl", "rb"))
#common_corpus = pickle.load(open(model_file+"_corpus.pkl", "rb"))
lda = LdaModel.load(model_file)

n_topics = lda.num_topics
print("Topics:", n_topics)

print("\nInferring topic mixture of unseen documents")
start_year = 1854
end_year = 1918
corpus_years = range(start_year, end_year)

for current_year in corpus_years:
    print("\nYear:", current_year)
    filename = "results/lda_new_doc_topics_" + str(current_year) + ".pkl"
    if not os.path.exists(filename):
        # create corpus of unseen documents
        articles_year = get_nlf_articles_for_year(year=current_year)
        print("Articles:", len(articles_year))
        print("Article:", articles_year[0])
        new_corpus = [common_dictionary.doc2bow(art) for art in articles_year]
        lda_doc_topics = [lda.get_document_topics(doc) for doc in new_corpus]
        #save inferred topics
        pickle_file = open(filename, 'wb')
        pickle.dump(lda_doc_topics, pickle_file)
        pickle_file.close()
        print("Saved doc topics as", filename)


# years = [1856, 1857, 1858, 1859, 1860]
# for current_year in years:
#     pickle_file = 'results/lda_new_doc_topics_'+str(current_year)+'.pkl'
#     print("\nYear:", current_year)
#     articles_year = get_nlf_articles_for_year(year=current_year)
#     print("Articles:", len(articles_year))
#     data = pickle.load(open(pickle_file,'rb'))
#     for i,doc in enumerate(data):
#         for k in doc:
#             if k[0] == 6 and k[1] > 0.2:
#                 print("Topic", str(k[0]+1) + " - share:", k[1])
#                 print(articles_year[i])
#                 print("--------------")
