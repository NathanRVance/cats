#!/usr/bin/env python3

# DO NOT UNCOMMENT THIS!!!
#import skynet

# Sorry, with such a rediculous name as perceptron, I couldn't resist.

import subprocess
import re
import dill

class Scorecard:
    # convert = lambda rent, child: entry
    # rent = preceeding word/tag pair
    # child = current word/tag pair
    # entry = hashable ID for this pair
    def __init__(self, convert):
        self.scores = {}
        self.convert = convert

    # Not actually logprobs, but score. Named as such to avoid modifying viterbi.
    def logprob(self, rent, child):
        entry = self.convert(rent, child)
        return self.__getScore(entry)

    def __getScore(self, entry):
        if entry not in self.scores:
            self.scores[entry] = 0
        return self.scores[entry]

    def adjust(self, rent, child, amount):
        entry = self.convert(rent, child)
        self.scores[entry] = self.__getScore(entry) + amount

class Perceptron:
    def __init__(self, disableMods):
        self.mods = [Scorecard(lambda rent, child: (rent.tag, child.tag)), # analagous to tagHmm
                Scorecard(lambda rent, child: (child.tag, child.word)), # analagous to wordHmm
                ]
        if not disableMods:
            self.mods.extend([Scorecard(lambda rent, child: (rent.word, child.tag)), # tag following word
                Scorecard(lambda rent, child: (rent.tag, child.word)), # word following tag
                Scorecard(lambda rent, child: (child.tag, len(child.word) == 1 and child.word[0] == child.word[0].upper() or len(child.word) > 1 and child.word[0] == child.word[0].upper() and child.word[1] == child.word[1].lower())), # SUPER ugly lambda for tag given uppercase first character with lowercase second
                Scorecard(lambda rent, child: (child.tag, 'http' in child.word or 'www' in child.word or '.com' in child.word or '.org' in child.word or '.net' in child.word)), # Is URL
                ])
        self.validTags = set('O')

    def learn(self, trainSents, trainAns, testSents, testAns):
        epocNum = 1
        prevFB1 = 0
        bestFB1 = 0
        bestEpoc = 0
        while True:
            print('Epoc {}:'.format(epocNum))
            print('Accuracy when training: {}'.format(self.runEpoc(trainSents, trainAns, True)))
            testAccuracy = self.runEpoc(testSents, testAns, False)
            scores = subprocess.check_output('perl conlleval.pl < perceptron.out', shell=True).decode('utf-8')
            FB1 = float(re.search(r'(?<=FB1:).+', scores).group(0).strip())
            print('Accuracy when testing: {}, with FB1: {}'.format(testAccuracy, FB1))
            print(scores)
            if FB1 > bestFB1:
                bestFB1 = FB1
                bestEpoc = epocNum
                print('New best score, pickling result to perceptron.pkl')
                with open('perceptron.pkl', 'wb') as f:
                    dill.dump(self, f)
            if bestEpoc < epocNum - 5: # It's been 5 epocs since we've had an improvement in FB1
                print('Detected overfitting, stopping at epoc {} with test FB1 {}'.format(bestEpoc, bestFB1))
                exit()
            print('So far, best score: {} At epoc: {}'.format(bestFB1, bestEpoc))
            epocNum += 1
            prevFB1 = FB1

    def runEpoc(self, sentances, answers, train):
        import viterbi
        total = 0
        correct = 0
        perceptOut = []
        for i, sent in enumerate(sentances):
            if len(sent) == 0:
                continue
            ans = answers[i]
            guess = viterbi.guessTags(sent, self.mods, self.validTags)
            perceptOut.append('{} {} {}'.format(sent[0], ans[0], guess[0]))
            for j in range(1, len(sent)):
                total += 1
                if ans[j] == guess[j]:
                    correct += 1
                perceptOut.append('{} {} {}'.format(sent[j], ans[j], guess[j]))
                if train:
                    self.validTags.add(ans[j]) # Adjust valid tags
                    parent = type("", (), {})() # Awful syntax for creating an anonymous class, but that's how you do it :(
                    parent.word = sent[j-1]
                    parent.tag = ans[j-1]
                    child = type("", (), {})()
                    child.word = sent[j]
                    child.tag = ans[j]
                    child.word = sent[j]
                    guessParent = type("", (), {})()
                    guessParent.word = sent[j-1]
                    guessParent.tag = guess[j-1]
                    guessChild = type("", (), {})()
                    guessChild.word = sent[j]
                    guessChild.tag = guess[j]
                    for mod in self.mods:
                        mod.adjust(parent, child, 1)
                        mod.adjust(guessParent, guessChild, -1)
            perceptOut.append('')
        with open('perceptron.out', 'w') as f:
            f.write('\n'.join(perceptOut))
        return correct / total
