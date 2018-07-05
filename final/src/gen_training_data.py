import pickle

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.preprocessing.sequence import pad_sequences

from src.data_loader import DataLoader


def pad(sentences, mapping, pad_length):
    return pad_sequences([[mapping.get(word, 0) for word in sentence] for sentence in sentences],
                         pad_length, padding='post', truncating='post')


def main(args):
    data_loader = DataLoader('raw_data/training_data/', 'raw_data/testing_data.csv')
    print('data loaded')
    train_pos = data_loader.gen_positive('training')
    print('train positive done')
    train_neg = data_loader.gen_negative('training')
    print('train negative done')
    valid_pos = data_loader.gen_positive('validation')
    print('valid positive done')
    valid_neg = data_loader.gen_negative('validation')
    print('valid negative done')
    with open('raw_data/train.pkl', 'wb') as fp:
        pickle.dump((train_pos, train_neg, valid_pos, valid_neg), fp)
    with open(args.training_path, 'rb') as fp:
        train_pos, train_neg, valid_pos, valid_neg = pickle.load(fp)
    with open(args.mapping, 'rb') as fp:
        mapping = pickle.load(fp)
    print('padding start')
    train_q = pad(train_pos[0] + train_neg[0], mapping, 71)
    print('train q done')
    train_a = pad(train_pos[1] + train_neg[1], mapping, 44)
    print('train a done')
    train_y = train_pos[2] + train_neg[2]
    print('train y done')

    valid_q = pad(valid_pos[0] + valid_neg[0], mapping, 71)
    print('valid q done')
    valid_a = pad(valid_pos[1] + valid_neg[1], mapping, 44)
    print('valid a done')
    valid_y = valid_pos[2] + valid_neg[2]
    print('valid y done')
    with open('raw_data/train_padded.pkl', 'wb') as fp:
        pickle.dump((train_q, train_a, train_y, valid_q, valid_a, valid_y), fp)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser('hw5')
    parser.add_argument('-train', '--training-path', default='./raw_data/train.pkl')
    parser.add_argument('-test', '--testing-path', default='./raw_data/testing_data.txt')
    parser.add_argument('--no-label-path', default='./raw_data/training_nolabel.txt')
    parser.add_argument('-o', '--output-path', default='prediction.csv')
    parser.add_argument('-m', '--model', default='stacked_gru_2_dot')
    parser.add_argument('--mapping', default='./models/mapping.pkl')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
