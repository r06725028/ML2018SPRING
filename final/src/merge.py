#!/usr/bin/env python
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.models import Model, load_model
from keras.layers import Input, add, multiply


def main(args):
    models = [rename(load_model(model_name), idx) for idx, model_name in enumerate(args.model_name)]
    q_input = Input(shape=(71,))
    a_input = Input(shape=(44,))
    outputs = globals().get(args.mode)([model([q_input, a_input]) for model in models])
    ensemble_model = Model([q_input, a_input], outputs)
    ensemble_model.save(f"models/{args.mode}_{'-'.join('_'.join(os.path.basename(model_name).split('_')[2:5]) for model_name in args.model_name)}")


def rename(model, idx):
    model.name = 'model_' + str(idx)
    return model


def parse_args():
    import argparse
    parser = argparse.ArgumentParser('final')
    parser.add_argument('-m', '--model_name', nargs='+')
    parser.add_argument('-mode', default='add')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())

