#!/usr/bin/env python3
import nltk
import re
import dill
import viterbi
import string
import cluster

class Categorizer():
    def __init__(self, topicsName, catsName):
        if topicsName == 'Baseline':
            self.topics = Baseline()
        elif topicsName == 'Perceptron':
            self.topics = Perceptron()
        else:
            self.topics = Everything()

        if catsName == 'BaseCategorizer':
            self.cats = BaseCategorizer()
        else:
            self.cats = ClusterCategorizer()

    def categorize(self, text):
        return self.cats.categorize(self.topics.topics(text))

def getLabels(cats):
    withLabels = {}
    for cat in cats:
        labelCounts = {}
        for topicList in cats[cat]:
            for label in topicList.split(' '):
                if label not in labelCounts:
                    labelCounts[label] = 0
                labelCounts[label] += 1
        bestLabel = max(labelCounts, key=lambda x: labelCounts[x])
        if bestLabel in withLabels:
            pass
            #print('Warning, appending to existing label {}!'.format(bestLabel))
        else:
            withLabels[bestLabel] = []
        for topicList in cats[cat]:
            withLabels[bestLabel].append(topicList)
    return withLabels

class BaseCategorizer():
    def __init__(self):
        self.cats = {}

    def categorize(self, important):
        # Decrement stale categories
        emptyCats = set()
        for cat in self.cats:
            if self.cats[cat] > 0:
                self.cats[cat] -= 1
            else:
                emptyCats.add(cat)
        for cat in emptyCats:
            del self.cats[cat]
        # Find categories
        bestCat = None
        for word in important:
            if word not in self.cats:
                self.cats[word] = 0
            self.cats[word] += 10
            if not bestCat or self.cats[bestCat] < self.cats[word]:
                bestCat = word
        return bestCat

class ClusterCategorizer():
    def __init__(self):
        self.seen = []
        self.boot = BaseCategorizer() # Used for the first few tweets
        cluster.init(bagOfWords = True)

    def categorize(self, topics):
        self.seen.append(' '.join(topics))
        if len(self.seen) > 100:
            self.seen = self.seen[1:]
        cluster.updateLanguage(self.seen[-1].split(' '))
        if len(self.seen) < 10:
            return self.boot.categorize(topics)
        cats = getLabels(cluster.cluster(self.seen))
        for cat in cats:
            if self.seen[-1] in cats[cat]:
                return cat
        return 'I dunno'

class Baseline():
    def __init__(self):
        self.remove = re.compile(r'http\S+|@|#|^rt ')

    def topics(self, text):
        text = self.remove.sub('', text)
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        topics = set()
        for tag in tagged:
            if tag[1] == 'NNP' or tag[1] == 'NN' and tag[0] in tokens:
                topics.add(tag[0])
        return topics

class Everything():
    def __init__(self):
        self.remove = re.compile(r'http\S+|@|#|^rt ')

    def topics(self, text):
        text = self.remove.sub('', text)
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        topics = set()
        for tag in tagged:
            if tag[0] not in string.punctuation:
                topics.add(tag[0])
        return topics

class Perceptron():
    def __init__(self):
        with open('perceptron.pkl', 'rb') as f:
            self.perceptron = dill.load(f)
        self.baseline = Baseline() # as a fallback

    def topics(self, text):
        sentance = text.split()
        tags = viterbi.guessTags(sentance, self.perceptron.mods, self.perceptron.validTags)
        topics = set()
        current = ''
        for i, tag in enumerate(tags):
            if tag[0] != 'I' and current:
                topics.add(current)
                current = ''
            elif tag[0] == 'B':
                current = sentance[i]
            elif tag[0] == 'I':
                current += ' ' + sentance[i]
        if not topics:
            topics = self.baseline.topics(text)
        return topics
