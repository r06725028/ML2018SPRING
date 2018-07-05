#!/usr/bin/env python
import pickle
import os

import numpy as np
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

from src import get_model


def pad(sentences, mapping, pad_length):
    return pad_sequences([[mapping.get(word, 0) for word in sentence] for sentence in sentences],
                         pad_length, padding='post', truncating='post')

def main(args):
    with open(args.training_path, 'rb') as fp:
        train_q, train_a, train_y, valid_q, valid_a, valid_y = pickle.load(fp)
    embedding = np.load('models/char_base_100_embedding.npy')
    model = get_model(args.model, embedding)
    os.makedirs(f'models/{args.model}', exist_ok=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = ModelCheckpoint(f'models/{args.model}/{args.model}' + '_{epoch:03d}_{val_loss:.4f}')
    plot_model(model, f'models/{args.model}/structure.png')
    model.fit([train_q, train_a], train_y, validation_data=([valid_q, valid_a], valid_y),
              batch_size=256, epochs=100, callbacks=[early_stopping, checkpoint])


def parse_args():
    import argparse
    parser = argparse.ArgumentParser('hw5')
    parser.add_argument('-train', '--training-path', default='./raw_data/train_padded.pkl')
    parser.add_argument('-test', '--testing-path', default='./raw_data/testing_data.txt')
    parser.add_argument('-o', '--output-path', default='prediction.csv')
    parser.add_argument('-m', '--model', default='stacked_gru_2_dot')
    parser.add_argument('--mapping', default='./models/mapping.pkl')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
