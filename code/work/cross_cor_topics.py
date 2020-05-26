import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt

def crosscorr(datax, datay, lag=0):
    return datax.corr(datay.shift(lag))

def best_corr(datax, datay, lag_range):
    corr_with_lag = [(lag, crosscorr(datax, datay, lag)) for lag in lag_range]
    return sorted(corr_with_lag,
                  key = lambda x: (abs(x[1]), -abs(x[0]), x[0]),
                  reverse=True)[0]

def topic_correlations(topic, cors):
    cors = [(e[0]+1, e[1]) for e in enumerate(cors[topic-1,:])]
    return sorted(cors, key=lambda x: abs(x[1]), reverse=True)

def topic_correlations_with_lag(topic, cors, lags):
    cors = [(e[0]+1, e[1], l) for e,l in zip(enumerate(cors[topic-1,:]),lags[topic-1,:])]
    return sorted(cors, key=lambda x: abs(x[1]), reverse=True)
    

def print_topic_correlations(file_name, topic_cors, topic_words, lag=False):
    with open(file_name, 'w') as out:
        for tc in topic_cors:
            t = tc[0]
            c = tc[1]
            l = tc[2] if lag else None               
            if c==1:
                print("TOPIC %d" %t, file=out)
                print(topic_words[t], file=out)
                print("", file=out)
                print("Correlations", file=out)
                print("", file=out)
            elif abs(c) < 0.5:
                break
            else:
                print("%d\t%2.4f" %(t,c), file=out)
                if lag:
                    print("lag: %d" %l, file=out)
                print(topic_words[t], file=out)
                print("", file=out)


def plot_topic_correlations(file_name, topic_cors, ts, topic_words):
    topic_cors = topic_cors[:7]
    positives = [tc for tc in topic_cors[1:] if tc[1]>0]
    negatives = [tc for tc in topic_cors[1:] if tc[1]<0]
    the_topic = topic_cors[0][0]
    labels = []
    
    fig = plt.figure(figsize=(15, 10))
    sns.lineplot(data=ts[[the_topic]], palette="gray_r", linewidth=6, legend=False)
    labels.append(str(the_topic) + " " + topic_words[the_topic])
    
    if positives:
        sns.lineplot(data=ts[[p[0] for p in positives]], hue="coherence", style="choice", palette="Reds", linewidth=2.5, legend=False)
        labels.extend(["%d %2.2f %s" %(p[0], p[1], topic_words[p[0]]) for p in positives])
    if negatives:
        sns.lineplot(data=ts[[n[0] for n in negatives]], hue="coherence", style="choice", palette="Blues", linewidth=2.5, legend=False)
        labels.extend(["%d %2.2f %s" %(n[0], n[1], topic_words[n[0]]) for n in negatives])

    plt.legend(labels = labels) 
    plt.savefig(file_name) 
                
if __name__ == "__main__":
    inp_file = sys.argv[1]
    shift = int(sys.argv[2])
    
    ts = pd.read_csv(inp_file, sep='\t')
    topic_words = pd.Series(ts.topic_words.values, index=ts.topic_id).to_dict()
    ts = ts.pivot(index='year', columns='topic_id', values='topic_weight')

    lag_range = range(-shift,shift)
    t_no = len(ts.columns)
    
    cors = np.ones((t_no, t_no))

    if shift == 0:
        for x in ts.columns[:-1]:
            for y in ts.columns[x:]:
                cors[x-1,y-1] = ts[x].corr(ts[y])
                cors[y-1,x-1] = ts[x].corr(ts[y])

        for t in [3, 6, 18, 22, 37, 42]:
            topic_cors = topic_correlations(t, cors)
            file_name = "correlations_" + str(t)
            print_topic_correlations(file_name+".txt", topic_cors, topic_words)
            plot_topic_correlations(file_name+".png", topic_cors, ts, topic_words)

                
    else:
        lags = np.zeros((t_no, t_no))
    
        for x in ts.columns[:-1]:
            for y in ts.columns[x:]:
                lag, cor =  best_corr(ts[x], ts[y], lag_range)
                cors[x-1,y-1] = cor
                cors[y-1,x-1] = cor
                lags[x-1,y-1] = lag
                lags[y-1,x-1] = -lag

        for t in [3, 6, 18, 22, 37, 42]:
            topic_cors = topic_correlations_with_lag(t, cors, lags)
            file_name = "correlations_w_lag_" + str(t)
            print_topic_correlations(file_name+".txt", topic_cors, topic_words, lag=True)
            plot_topic_correlations(file_name+".png", topic_cors, ts, topic_words)
            
                
