# This code is heavily modified version of the code presented in the paper "Unsupervised Neural Machine Translation" by Mikel artetxe





SPECIAL_SYMBOLS = 5
PAD, EOS, SEP1, SEP2, OOV = 0, 1, 2, 3, 4   #REC=start reconstruction in backtranslation and denoising
                                     #SUMMARY=start the summary

class Dictionary:
    def __init__(self, model):
        self.id2word = model.index2word
        self.word2id = {word: i+SPECIAL_SYMBOLS for i, word in enumerate(self.id2word)}
        self.word2id["<sep1>"]=SEP1
        self.word2id["<sep2>"]=SEP2


    def sentence2ids(self, sentence, eos=False):
        tokens = tokenize(sentence)
        ids = [self.word2id[word] if word in self.word2id else OOV for word in tokens] #needs to be edited
        if eos:
            ids = ids + [EOS]
        return ids

    def sentences2ids(self, sentences, eos=False):
        ids = [self.sentence2ids(sentence, eos=eos) for sentence in sentences]
        lengths = [len(s) for s in ids]
        ids = [s + [PAD]*(max(lengths)-len(s)) for s in ids]  # Padding
        ids = [[ids[i][j] for i in range(len(ids))] for j in range(max(lengths))]  # batch*len -> len*batch
        return ids, lengths


    def size(self):
        return len(self.id2word)


def special_ids(ids):
    return (ids) * (ids < SPECIAL_SYMBOLS).long()


def word_ids(ids):
    return (ids-SPECIAL_SYMBOLS) * (ids >= SPECIAL_SYMBOLS).long()

def tokenize(sentence):
    return sentence.strip().split()
