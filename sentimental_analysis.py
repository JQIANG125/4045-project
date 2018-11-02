import json
import nltk
from functools import reduce
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
from multiprocessing import Pool
from datetime import datetime


POSITIVE = 0
NEGATIVE = 1
LAPLACE_SMOOTH_VALUE = 5
STOPWORDS = set(stopwords.words('english') + list(string.punctuation))


def __treebank_to_wordnet__(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return tag


def __preprocess__(sentence, rating):
    lemmatizer = WordNetLemmatizer()
    whitelist = ['UH', 'VB', 'VBZ', 'VBP', 'VBD', 'VBN', 'VBG', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'RP']
    sentences = nltk.sent_tokenize(sentence)
    words = []

    regex = re.compile('^[A-Z]+\$?$')
    for sentence in sentences:
        for w in nltk.word_tokenize(sentence):
            w = w.lower()
            if w not in words and w not in STOPWORDS:
                for _, tag in nltk.pos_tag(w):
                    if regex.match(tag) and tag in whitelist:
                        w_ = lemmatizer.lemmatize(w, pos=__treebank_to_wordnet__(tag))
                        if w_ not in words:
                            words.append(w_)

    return words, rating


if __name__ == '__main__':

    mp_pool = Pool(processes=3)
    word_map = {None: [0, 0]}
    prob_map = {}

    start = datetime.now()
    with open("CellPhoneReview.json") as datafile:
        data = datafile.read()

    data = ("[%s]" % data).replace('}\n{', '},{')
    data = json.loads(data)
    data = mp_pool.starmap(__preprocess__, [(datum['reviewText'], datum['overall']) for datum in data])

    for datum in data:
        pos_rating = datum[1]
        neg_rating = 5 - pos_rating + 1
        for word in datum[0]:
            if word not in word_map:
                word_map[word] = [LAPLACE_SMOOTH_VALUE, LAPLACE_SMOOTH_VALUE]
            word_map[word][POSITIVE] = word_map[word][POSITIVE] + pos_rating
            word_map[word][NEGATIVE] = word_map[word][NEGATIVE] + neg_rating
            word_map[None][POSITIVE] = word_map[None][POSITIVE] + pos_rating
            word_map[None][NEGATIVE] = word_map[None][NEGATIVE] + neg_rating

    word_map[None] = [word_map[None][idx] + LAPLACE_SMOOTH_VALUE * len(word_map.keys())
                      for idx in range(len(word_map[None]))]
    print(word_map[None])
    total_count = reduce(lambda a, b: a + b, word_map[None])
    prob_map[None] = [0, 0]
    for idx in range(len(word_map[None])):
        prob_map[None][idx] = word_map[None][idx] / total_count

    for x in word_map:
        if x is not None:
            prob_word = reduce(lambda a, b: a + b, word_map[x]) / total_count
            prob_map[x] = [0, 0]
            for idx in range(len(word_map[x])):
                # P(Word|Emotion)
                prob_map[x][idx] = word_map[x][idx] / word_map[None][idx]

    del prob_map[None]
    sorted_words = sorted(prob_map.keys(),
                          key=lambda a: prob_map[a][POSITIVE] - prob_map[a][NEGATIVE],
                          reverse=True)
    print("\nPositive-Words")
    for word in sorted_words[:20]:
        print("%s: %s" % (word, prob_map[word][POSITIVE]))

    sorted_words = sorted(prob_map.keys(),
                          key=lambda a: prob_map[a][NEGATIVE] - prob_map[a][POSITIVE],
                          reverse=True)
    print("\nNegative-Words")
    for word in sorted_words[:20]:
        print("%s: %s" % (word, prob_map[word][NEGATIVE]))

    print("Completed calculations in %d seconds" % (datetime.now() - start).total_seconds())
