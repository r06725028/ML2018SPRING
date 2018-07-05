#!/usr/bin/env python
import pickle
import os

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.models import load_model
import pandas as pd


def main(args):
    model = load_model(args.model)
    print('model loaded')
    with open(args.valid_path, 'rb') as fp:
        valid_id, valid_q, valid_a, valid_y = pickle.load(fp)
    valid_id = list(range(224400)) + [x // 5 for x in range(224400 * 5)]
    print(args.model)
    print('data loaded')
    predictions = model.predict([valid_q, valid_a], batch_size=1024).ravel()
    print('predicted')
    prediction_df = pd.DataFrame({'id': valid_id, 'option': valid_y, 'score': predictions})
    prediction_df.to_csv(f'predictions/valid/{os.path.basename(args.model)}_valid.csv', index=False)
    prediction_s = prediction_df.groupby('id').apply(lambda df: df.set_index('option').idxmax())['score']
    print('decided')
    print('score:', prediction_s.mean())


def parse_args():
    import argparse
    parser = argparse.ArgumentParser('hw5')
    parser.add_argument('-valid', '--valid-path', default='./raw_data/valid_padded.pkl')
    parser.add_argument('-o', '--output-path', default='prediction.csv')
    parser.add_argument('-m', '--model', default='stacked_gru_2_dot')
    parser.add_argument('--mapping', default='./models/mapping.pkl')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
