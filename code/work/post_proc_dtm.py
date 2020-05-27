from gensim.models import LdaSeqModel
import pickle
import os

from corpus import get_nlf_articles_for_year
from infer_topics import get_new_doc_topics

model_dir = "trained_models/dtm/"
model_file = model_dir + "ldaseq100_nlf_50topics_64_years"
print("Opening trained TM at", model_file)

common_dictionary = pickle.load(open(model_file+"_dict.pkl", "rb"))
common_corpus = pickle.load(open(model_file+"_corpus.pkl", "rb"))
ldaseq = LdaSeqModel.load(model_file)

n_topics = len(ldaseq.print_topics(0))
n_timeslices = len(ldaseq.time_slice)
rev_dictionary = {common_dictionary[k]:k for k in common_dictionary.keys()}
print("Topics:", n_topics)
print("Timeslices:", n_timeslices)

print("\nInferring topic mixture of unseen documents")
start_year = 1854
end_year = 1918
corpus_years = range(start_year, end_year)

for current_year in corpus_years:
    print("\nYear:", current_year)
    filename = "results/new_doc_topics_" + str(current_year) + ".pkl"
    if not os.path.exists(filename):
        # create corpus of unseen documents
        articles_year = get_nlf_articles_for_year(year=current_year)
        print("Articles:", len(articles_year))
        # infer topic mixture of unseen documents
        time_slice = current_year - start_year
        doc_topics = get_new_doc_topics(model=ldaseq, new_documents=articles_year, time_slice=time_slice)
        #save inferred topics for unseen documents
        pickle_file = open(filename, 'wb')
        pickle.dump(doc_topics, pickle_file)
        pickle_file.close()
        print("Saved doc topics as", filename)


