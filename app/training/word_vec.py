import gensim
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()


class Word2Vec:

    word2vecFile = os.getenv("GOOGLE_VECTORS_LOCATION")
    w2v = ""

    def loadvec(self):
        print("loading vectors  . . .")
        Word2Vec.w2v = gensim.models.KeyedVectors.load_word2vec_format(Word2Vec.word2vecFile, binary=True)
        print("vectors ready")

    def get_vector(self, word):
        wordvec = []
        try:
            wordvec = Word2Vec.w2v[word]
        except:
            wordvec = np.zeros(300)

        return wordvec
