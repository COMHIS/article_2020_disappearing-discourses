from corpus get_nlf_articles_ads_supplements
from Collections import defaultdict
import pickle

articles, ads, sup = get_nlf_articles_ads_supplements(start_year=start_1854, end_year=1917, min_article_len=20)

ttr = {}

for y in sorted(articles.keys()):
    print(y)
    types_tokens = defaultdict(int)
    for article in articles[y]:
        for w in article.lower().split():
            types_tokens[w] += 1

    ttr = len(types_tokens.keys()) / sum(list(types_tokens.values()))
    print(ttr)

    ttr[y] = ttr

pickle.dump(ttr, open("ttr.pkl", "wb"))
