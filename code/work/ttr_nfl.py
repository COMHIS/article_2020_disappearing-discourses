from corpus import get_nlf_articles_ads_supplements
from collections import defaultdict
import pickle

articles, ads, sup = get_nlf_articles_ads_supplements(start_year=1854, end_year=1917, min_article_len=20)

ttr = {}

for y in sorted(articles.keys()):
    print(y)
    types_tokens = defaultdict(int)
    for article in articles[y]:
        try:
            for w in article:
                types_tokens[w.lower()] += 1
        except Exception as e:
            print("ARTICLE: %s" %article)
            raise e

    TTR = len(types_tokens.keys()) / sum(list(types_tokens.values()))
    print(TTR)

    ttr[y] = TTR

pickle.dump(ttr, open("ttr.pkl", "wb"))
