import pickle

import numpy as np

from gensim.models import Word2Vec


def main():
    word2vec = Word2Vec.load('models/char_base_100')
    mapping = {word: idx + 1 for idx, word in enumerate(word2vec.wv.index2word)}
    embedding = np.concatenate([np.zeros((1, word2vec.wv.syn0.shape[1])), word2vec.wv.syn0])
    
    with open('models/mapping.pkl', 'wb') as fp:
        pickle.dump(mapping, fp)
    np.save('models/char_base_100_embedding.npy', embedding)


if __name__ == '__main__':
    main()
