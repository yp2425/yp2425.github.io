import re
import os
import pickle
import numpy as np
from nltk import ngrams
from collections import Counter
import getdata
from getdata import getobject

train_data= getobject()
class Autocomplete():
    
    def __init__(self, model_path="./", sentences=train_data, n_model=3, n_candidates=10, match_model="middle",
                 min_freq=5, punctuations="""!"#$%&\'()*+,./:;<=>?@[\\]^_{|}~""", lowercase=True):
        # Model parameters
        self.n_model = n_model
        # number of candidates suggested sentences to show
        self.n_candidates = n_candidates
        # path to the folder that stores the language model
        self.model_path = model_path
        # type of autocomplete model
        # `start`, `end` of `middle`
        self.match_model = match_model
        # not consider ngrams that appear less than this value 
        self.min_freq = min_freq
        # punctuations to remove
        self.punctuations = punctuations
        # lowercase the sentences?
        self.lowercase = lowercase
        # list of sentences to use to train the model
        if sentences is None:
            sentences = []
        self.sentences = sentences

        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        # loading the language model
        for N in range(1, self.n_model + 1):
            filename = self.model_path + "/" + str(N) + "-grams.pickle"
            if not os.path.exists(filename):
                # if no language model is found, then it is computed
                # remove the dashes and the bendy apostrophe
                if not self.sentences:
                    raise Exception("You need to give a sample sentences to train the model!")
                self.compute_language_model()

        # ngrams_freqs is a dictionary whose keys are the ngrams labels and the values their counts
        self.ngrams_freqs = dict()
        for N in range(1, self.n_model + 1):
            filename = self.model_path + "/" + str(N) + "-grams.pickle"
            with open(filename, "rb") as f:
                self.ngrams_freqs[N] = pickle.load(f)

        # saving the ngrams_freqs keys in a separate dictionary
        self.ngrams_keys = dict()
        for N in range(1, self.n_model + 1):
            self.ngrams_keys[N] = list(self.ngrams_freqs[N].keys())

        # saving the total counts
        self.total_counts = [sum(self.ngrams_freqs[N].values()) for N in range(1, self.n_model + 1)]


    def get_ngrams(self, sentence, n=1):
        """
        Given a sentence returns a list of its n-grams
        """
        # remove punctuation
        if self.punctuations != "":
            sentence = re.sub('[' + self.punctuations + ']', ' ', sentence).strip()
        if self.lowercase:
            sentence = sentence.lower()
        # generate tokens
        if n > 1:
            sentence = [" ".join(n) for n in ngrams(sentence.split(), n, pad_right=True, right_pad_symbol='</END>')]
        else:
            sentence = sentence.split()
        # filter for empty string
        return list(filter(None, sentence))


    def compute_language_model(self):
        """
        Given a list of sentences compute the n-grams
        """
        #if len(self.sentences) < 1e4:
        for N in range(1, self.n_model + 1):
            ngrams_list = []
            for sentence in self.sentences:
                ngrams_sentence = self.get_ngrams(sentence, n=N)
                ngrams_list.extend(ngrams_sentence)
            ngrams_freqs = Counter(ngrams_list)
            filename = self.model_path + "/" + str(N) + "-grams.pickle"
            with open(filename, "wb") as f:
                pickle.dump(ngrams_freqs, f,protocol=2)
            print("Saving the %s-grams in %s" % (N, filename))


    def compute_prob_sentence(self, sentence):
        """
        Given a sentence, return the log probability of that sentence using the n-gram approximation
        :return:
        """
        if sentence != "":
            total_prob = 0
            pieces = sentence.split()
            for i in range(1, len(pieces) + 1):
                if i <= self.n_model:
                    piece = pieces[:i]
                else:
                    piece = pieces[i - self.n_model:i]
                #
                ngram_model_to_use = len(piece)
                piece_lbl = " ".join(piece)
                if ngram_model_to_use in self.ngrams_freqs:
                    den = float(self.total_counts[ngram_model_to_use - 1])
                    num = float(self.ngrams_freqs[ngram_model_to_use].get(piece_lbl.lower(), 0))
                    piece_prob = np.log10(num/den)
                else:
                    return -np.inf
                total_prob += piece_prob
            return total_prob
        else:
            return -100


    def predictions(self, word):
        
        word = word.lower()
        parts = word.split()
        beginning = ""
        if len(parts) >= self.n_model:
            beginning = " ".join(parts[:-self.n_model + 1])
            word = " ".join(parts[-self.n_model + 1:])
        #
        if self.match_model == "start":
            candidates = np.array(list(filter(lambda x: x.startswith(word), self.ngrams_keys.get(self.n_model, ''))))
        elif self.match_model == "end":
            candidates = np.array(list(filter(lambda x: x.endswith(word), self.ngrams_keys.get(self.n_model, ''))))
        elif self.match_model == "middle":
            candidates = np.array(list(filter(None, [key if word in key else None for key in self.ngrams_keys.get(self.n_model, '')])))[::-1]
        else:
            raise Exception("match_model can only be `start`, `end` or `middle`")
        #
        if len(candidates) == 0:
            return [], []
        #
        predictions = []
        if len(candidates) >= 1:
            for i in range(len(candidates)):
                if beginning == "":
                    predictions.append(" ".join([beginning, candidates[i].replace("</END>", "").capitalize()]).strip())
                else:
                    predictions.append(" ".join([beginning.capitalize(), candidates[i].replace("</END>", "")]).strip())
        #
        predictions = np.array(predictions)
        probabilities = np.array(
            [self.compute_prob_sentence(sentence) for sentence in predictions])
        order = np.argsort(probabilities)[::-1]
        predictions = list(predictions[order][:self.n_candidates])
        probabilities = list(probabilities[order][:self.n_candidates])
        #
        return predictions



