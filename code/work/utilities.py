def get_key_val_pair(s):
    return(s.strip().split(" "))

def list_to_dict(d):
    new_dict = {}
    len_dict = len(d)
    for i in range(len_dict):
        key_val = get_key_val_pair(d[i])
        if key_val[0] in new_dict.keys():
            new_dict[key_val[0]].append(key_val[1])
        else:
            new_dict[key_val[0]] = [key_val[1]]
    return new_dict

def save_dict(d, filename):
    with open(filename, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

def load_dict(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def topic_to_word_cloud(vocab, topic_dist):
    from wordcloud import WordCloud

    # Build a dict of weights for each word
    word_weights = dict([
        (vocab.id2token[i], topic_dist[i]) for i in range(len(topic_dist))
    ])
    # Generate a word cloud for this topic
    wordcloud = WordCloud(
        width=480, height=480,
        background_color=None, mode="RGBA",
    ).generate_from_frequencies(word_weights)
    # Get the image (PIL)
    image = wordcloud.to_image()

    return image

