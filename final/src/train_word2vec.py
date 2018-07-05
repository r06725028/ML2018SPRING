#!/usr/bin/env python
import glob
import logging

from gensim.models import Word2Vec


def main(args):
    sentences = []
    for filename in glob.glob('./raw_data/training_data/*'):
        with open(filename, 'r') as fp:
            sentences.extend([list(line.strip()) for line in fp])
    word2vec = Word2Vec(sentences, size=args.size, min_count=1, workers=16, sg=int(args.sg))
    word2vec.save(f'./models/char_base_{args.size}_sg_{args.sg}')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser('train w2v')
    parser.add_argument('-s', '--size', default=100, type=int)
    parser.add_argument('-g', '--sg', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main(parse_args())
