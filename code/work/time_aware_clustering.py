import pandas as pd
import numpy as np
import sys
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt

def average_period(period, ts):
    return ts.loc[:, period[0]:period[-1]].mean(axis=1)

def distance_at_place_average(periods, place, ts):
    return jensenshannon(average_period(periods[place],ts), average_period(periods[place+1],ts))


def distance_at_place_pairwise(periods, place, ts):
    return np.mean([jensenshannon(ts[y1], ts[y2]) for y1 in periods[place] for y2 in periods[place+1]])

def distance_at_place_start_end_(periods, place, ts):
    # I don't like the result
    # beginning is still similar to end
    return jensenshannon(ts[periods[place][0]], ts[periods[place][-1]])

def distance_at_place(periods, place, ts):
    distance_at_place_pairwise(periods, place, ts)


def vizualize(cluster_table, periods):
    figure = plt.gcf()    
    figure.set_size_inches(16, 12)       

    plt.title('Time-aware Clustering Dendrogram')
    plt.xlabel('Date')
    plt.ylabel('distance')
    
    dendrogram(cluster_table, leaf_font_size=8, leaf_rotation=45, labels=[p[0] for p in periods])

    plt.tight_layout()   
    plt.savefig("dendrogram.png")


def time_aware_clustering(periods, ts):
    cluster_number = {p:i for i,p in enumerate(periods)}
    cluster_counter = len(periods)

    dists = [distance_at_place(periods, i, ts) for i in range(len(periods)-1)]

    cluster_table = None

    while len(periods) > 1:
        merge_left_index = np.argmin(np.array(dists))

        new_period = periods[merge_left_index]+periods[merge_left_index+1]
        
        new_cluster = [cluster_number[periods[merge_left_index]],
                       cluster_number[periods[merge_left_index+1]],
                       dists[merge_left_index],
                       len(new_period)]

        cluster_number[new_period] = cluster_counter
        cluster_counter += 1
        
        if cluster_table is None:
            cluster_table = np.array([new_cluster])
        else:
            cluster_table = np.vstack([cluster_table, np.array(new_cluster)])

        periods = periods[:merge_left_index] + [new_period] + periods[merge_left_index+2:]
        if len(periods) == 1:
            break
        
        new_dists = []
        for i,d in enumerate(dists):
            if i == merge_left_index+1:
                continue
            elif i in [merge_left_index-1, merge_left_index]:
                if i < len(periods)-1:
                    new_dists.append(distance_at_place(periods, i, ts))
            else:
                new_dists.append(d)
        dists = new_dists
                
            
    print("FINAL: ", cluster_table) 

    return cluster_table


if __name__ == "__main__":
    inp_file = sys.argv[1]
    ts = pd.read_csv(inp_file, sep='\t')
    topic_words = pd.Series(ts.topic_words.values, index=ts.topic_id).to_dict()
    ts = ts.pivot(index='topic_id', columns='year', values='topic_weight')
    
    periods = [(y,) for y in list(ts.columns)]
    cluster_table = time_aware_clustering(periods, ts)
    print(cluster_table)


    vizualize(cluster_table, periods)
    
