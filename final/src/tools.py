import pickle
import os

from keras.preprocessing.sequence import pad_sequences


def cache(func, filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)
    else:
        data = func()
        with open(filename, 'wb') as fp:
            pickle.dump(data, fp)
        return data


def pad(sentences, mapping, pad_length):
    return pad_sequences([[mapping.get(word, 0) for word in sentence] for sentence in sentences],
                         pad_length, padding='post', truncating='post')

def gen_training_data(training_path):
    data_loader = DataLoader(training_path, '')
    print('data loaded')
    train_pos = data_loader.gen_positive('training')
    print('train positive done')
    train_neg = data_loader.gen_negative('training')
    print('train negative done')
    valid_pos = data_loader.gen_positive('validation')
    print('valid positive done')
    valid_neg = data_loader.gen_negative('validation')
    print('valid negative done')
    return (train_pos, train_neg, valid_pos, valid_neg)


def pad_training_data(train_pos, train_neg, valid_pos, valid_neg):
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
    return (train_q, train_a, train_y, valid_q, valid_a, valid_y)


def gen_testing_data(testing_path):
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
    return test_q, test_a


def gen_word2vec_mapping(word2vec_path, mapping_path, embedding_path):
    word2vec = Word2Vec.load(word2vec_path)
    mapping = {word: idx + 1 for idx, word in enumerate(word2vec.wv.index2word)}
    embedding = np.concatenate([np.zeros((1, word2vec.wv.syn0.shape[1])), word2vec.wv.syn0])
    
    with open('models/mapping.pkl', 'wb') as fp:
        pickle.dump(mapping, fp)
    np.save('models/char_base_100_embedding.npy', embedding)
