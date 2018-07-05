import pickle

from keras.preprocessing.sequence import pad_sequences
import pandas as pd


def pad(sentences, mapping, pad_length):
    return pad_sequences([[mapping.get(word, 0) for word in sentence] for sentence in sentences],
                         pad_length, padding='post', truncating='post')


def main():
    test_df = pd.read_csv('raw_data/testing_data.csv')
    test_id, test_index, test_q, test_a = [], [], [], []
    with open('./models/mapping.pkl', 'rb') as fp:
        mapping = pickle.load(fp)
    for row in test_df.itertuples():
        for option in row.options.split('\t'):
            idx, opt = option.split(':', 1)
            test_id.append(row.id)
            test_index.append(idx)
            test_q.append(row.dialogue)
            test_a.append(opt)
    test_q = pad(test_q, mapping, 71)
    test_a = pad(test_a, mapping, 44)
    with open('raw_data/testing_data_extend.pkl', 'wb') as fp:
        pickle.dump((test_id, test_index, test_q, test_a), fp)


if __name__ == '__main__':
    main()
