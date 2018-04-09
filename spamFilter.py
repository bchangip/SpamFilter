from random import shuffle
import re
from collections import Counter
from itertools import chain
from functools import reduce
from operator import mul
from numpy import arange

class SpamFilter(object):
  def sanitizeMessage(self, message):
    # Remove non letters
    message = re.sub('[^a-z\s]', "", message.lower())

    # Remove double spaces
    message = re.sub('\s+', " ", message)

    # Remove starting spaces
    message = re.sub('^\s', "", message)

    # Remove trailing spaces
    return re.sub('\s$', "", message)

  def __init__(self, trainingFile, K):
    inputData = open(trainingFile).readlines()

    inputData = [message.strip().split('\t') for message in inputData]

    # Remove non letters
    inputData = [[message[0], self.sanitizeMessage(message[1])] for message in inputData]

    hams = list(filter(lambda x: x[0] == 'ham', inputData))
    spams = list(filter(lambda x: x[0] == 'spam', inputData))

    shuffle(hams)
    shuffle(spams)

    trainingHamIndex = int(len(hams)*0.8)
    trainingSpamIndex = int(len(spams)*0.8)

    validationHamIndex = trainingHamIndex + int(len(hams)*0.1)
    validationSpamIndex = trainingSpamIndex + int(len(spams)*0.1)

    trainingHams = hams[:trainingHamIndex]
    self.validationHams = hams[trainingHamIndex:validationHamIndex]
    self.crossValidationHams = hams[validationHamIndex:]
    with open('hams.txt', 'w') as hamsOutput:
      for ham in trainingHams:
        hamsOutput.write(ham[0]+'\t'+ham[1]+'\n')

    trainingSpams = spams[:trainingSpamIndex]
    self.validationSpams = spams[trainingSpamIndex:validationSpamIndex]
    self.crossValidationSpams = spams[validationSpamIndex:]
    # print('trainingSpams', trainingSpams)
    with open('spams.txt', 'w') as spamsOutput:
      for spam in trainingSpams:
        spamsOutput.write(spam[0]+'\t'+spam[1]+'\n')

    hamWords = " ".join([message[1] for message in trainingHams])
    self.hamFrequencies = Counter(hamWords.split(" "))
    self.hamLen = len(hamWords)

    spamWords = " ".join([message[1] for message in trainingSpams])
    self.spamFrequencies = Counter(spamWords.split(" "))
    self.spamLen = len(spamWords)

    self.hamSpamRatio = (len(trainingHams)/(len(trainingHams)+len(trainingSpams))) / (len(trainingSpams)/(len(trainingHams)+len(trainingSpams)))
    self.K = K


  def wordSpamProbability(self, word):
    return (self.spamFrequencies[word]+self.K) / (self.spamLen + self.K*len(self.spamFrequencies+self.hamFrequencies))

  def wordHamProbability(self, word):
    return (self.hamFrequencies[word]+self.K) / (self.hamLen + self.K*len(self.spamFrequencies+self.hamFrequencies))

  def messageSpamProbability(self, message):
    # print('message', message)
    splittedMessage = self.sanitizeMessage(message).split(' ')
    spamProbabilities = [self.wordSpamProbability(word) for word in splittedMessage]
    hamProbabilities = [self.wordHamProbability(word) for word in splittedMessage]
    spamReduced = reduce(mul, spamProbabilities, 1)
    # print('spamReduced', spamReduced)
    if spamReduced == 0:
      spamReduced = 1e-200
    hamReduced = reduce(mul, hamProbabilities, 1)
    # print('hamSpamRatio', self.hamSpamRatio)
    spamProbability = (1 + self.hamSpamRatio*hamReduced/spamReduced)**-1
    # print('spamProbability', spamProbability)
    if spamProbability > 0.7:
        return ('spam', spamProbability)
    else:
        return ('ham', spamProbability)

  def efficiency(self):
    hits = 0
    misses = 0
    for message in self.validationHams:
      if self.messageSpamProbability(message[1])[0] == 'ham':
        print('Correct ham')
        hits += 1
      else:
        print('Failed ham')
        misses += 1

    for message in self.validationSpams:
      if self.messageSpamProbability(message[1])[0] == 'spam':
        print('Correct spam')
        hits += 1
      else:
        print('Failed spam')
        misses += 1

    return hits/(hits+misses)


spamFilter = SpamFilter('corpus.txt', 0.01)
print(spamFilter.efficiency())

with open('inputMessages.txt', 'r') as inputMessages:
  inputMessages = inputMessages.readlines()

results = [spamFilter.messageSpamProbability(message)[0] + '\t' + message for message in inputMessages]

with open('outputMessages.txt', 'w') as outputMessages:
  outputMessages.write(results)
