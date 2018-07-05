#!/usr/bin/env python
import pickle

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.models import load_model
import pandas as pd


def main(args):
    model = load_model(args.model)
    with open('raw_data/testing_data_extend.pkl', 'rb') as fp:
        test_id, option_index, test_q, test_a = pickle.load(fp)
    predictions = model.predict([test_q, test_a], batch_size=256).ravel()
    prediction_df = pd.DataFrame({'id': test_id, 'option': option_index, 'score': predictions})
    prediction_s = prediction_df.groupby('id').apply(lambda df: df.set_index('option').idxmax())['score']
    prediction_s.name = 'ans'
    prediction_s.to_csv(args.output_path, header=True)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser('final')
    parser.add_argument('-train', '--training-path', default='./raw_data/train_padded.pkl')
    parser.add_argument('-test', '--testing-path', default='./raw_data/testing_data.txt')
    parser.add_argument('-o', '--output-path', default='prediction.csv')
    parser.add_argument('-m', '--model', default='stacked_gru_2_dot')
    parser.add_argument('--mapping', default='./models/mapping.pkl')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
