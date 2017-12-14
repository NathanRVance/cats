#!/usr/bin/env python3
import subprocess
import os
import sys
from sklearn.cluster import KMeans
import numpy
import math

def init(corpus = None, bagOfWords = False, language = []):
    global model
    if bagOfWords:
        #print('Using bag-of-words model')
        global word2index
        word2index = []
        for line in language:
            for word in line.split():
                if word not in word2index:
                    word2index.append(word)
        class Inferer:
            def infer_vector(self, words):
                vec = [0] * len(word2index)
                for word in words:
                    vec[word2index.index(word)] += 1
                return vec
        model = Inferer()
    else:
        import gensim
        print('Using doc2vec')
        currDir = os.path.dirname(os.path.realpath(__file__))
        savePath = currDir + '/doc2vec.model'
        if os.path.isfile(savePath):
            print('Loading doc2vec...')
            model = gensim.models.doc2vec.Doc2Vec.load(savePath)
        else:
            if not corpus:
                print('ERROR: Must include path to corpus if no saved model is present.')
                sys.exit(1)
            print('Training doc2vec...')
            def read_corpus(fname, tokens_only=False):
                with open(fname) as f:
                    for i, line in enumerate(f):
                        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])
            trainCorpus = list(read_corpus(corpus))
            model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
            model.build_vocab(trainCorpus)
            model.train(trainCorpus, total_examples=model.corpus_count, epochs=model.iter)
            model.save(savePath)
            print('Done training doc2vec.')

def updateLanguage(words):
    global word2index
    for word in words:
        if word not in word2index:
            word2index.append(word)

def getVec(text):
    global model
    return model.infer_vector(text.split())

def cluster(tweets):
    toCluster = numpy.array([getVec(tweet) for tweet in tweets])
    labels = None
    score = 0
    for i in range(2, int(math.log2(len(tweets)))):
        kmeans = KMeans(n_clusters=i).fit(toCluster)
        unique, counts = numpy.unique(kmeans.labels_, return_counts=True)
        # Score is percent off of even distribution 
        kmeansScore = min(counts) / (len(tweets) / i)
        if kmeansScore > score:
            score = kmeansScore
            labels = kmeans.labels_
        #print('Score for {} clusters was: {}'.format(i, kmeansScore))
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(tweets[i])
    #print('For {} tweets, detected {} clusters'.format(len(tweets), len(clusters)))
    return clusters
