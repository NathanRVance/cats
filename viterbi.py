# This is basically the Viterbi algorithm from HW2, with minor adjustments
def guessTags(sentance, mods, validTags):
    class WordTagPair:
        def __init__(self, word, tag):
            self.word = word
            self.tag = tag
            self.prob = -100000000 # probs small enough...
            self.bestRent = None
        def addParent(self, rent):
            rentProb = rent.prob
            for mod in mods:
                rentProb += mod.logprob(rent, self)
            if not self.bestRent or self.prob < rentProb:
                self.bestRent = rent
                self.prob = rentProb
        def backRecurse(self):
            if not self.bestRent: # Base case
                return []
            ret = self.bestRent.backRecurse()
            ret.append(self.tag)
            return ret

    initPair = WordTagPair('', '<s>')
    initPair.pairProb = 0
    initPair.prob = 0
    prevLayer = [initPair]
    curLayer = []
    for word in sentance:
        for tag in validTags:
            pair = WordTagPair(word, tag)
            for rent in prevLayer:
                pair.addParent(rent)
            curLayer.append(pair)
        prevLayer = curLayer
        curLayer = []
    best = max(prevLayer, key=lambda x: x.prob)
    return best.backRecurse()
