#!/usr/bin/env python3
import json
import argparse
import topic
from random import shuffle

parser = argparse.ArgumentParser(description='Test schemes for categorizing tweets')
parser.add_argument('tweets', nargs='+', help='file of tweets')
args = parser.parse_args()

tweets = {}
categories = {}

for name in args.tweets:
    tweets[name] = []
    with open(name) as f:
        first = True
        for line in f:
            if first:
                categories[name] = json.loads(line)['categories']
                first = False
            else:
                tweets[name].append(json.loads(line)['text'].lower())

def evaluate(topicsName, catsName):
    print('Evaluating {} topic extractor with {}'.format(topicsName, catsName))
    categorizer = topic.Categorizer(topicsName, catsName)
    baseCats = {}
    for tweet in testTweets:
        cat = categorizer.categorize(tweet)
        if cat not in baseCats:
            baseCats[cat] = set()
        baseCats[cat].add(tweet)

    correct = 0
    for cat in reversed(sorted(baseCats, key=lambda x: len(baseCats[x]))):
        print('{}: {}'.format(cat, len(baseCats[cat])))
        for tweet in baseCats[cat]:
            for name in tweets:
                if tweet in tweets[name]:
                    realName = name
            if cat in categories[realName]:
                correct += 1
    print('Got {} of {} correct'.format(correct, len(testTweets)))
    percent = correct / len(testTweets) * 100
    print('Score: {}%'.format(int(percent)))
    with open('out', 'a') as f:
        f.write('{} {} {} {} {} {} {}\n'.format(topicsName, catsName, correct, len(testTweets), percent, len(baseCats), len(tweets)))

for topicsName in ['Baseline']:#, 'Perceptron', 'Everything']:
    for catsName in ['BaseCategorizer', 'ClusterCategorizer']:
        for trial in range(0, 10):
            testTweets = []
            for cat in tweets:
                shuffle(tweets[cat])
                for tweet in tweets[cat][0:100]:
                    testTweets.append(tweet)
            shuffle(testTweets)
            evaluate(topicsName, catsName)
