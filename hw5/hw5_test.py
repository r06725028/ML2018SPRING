import sys, argparse, os
import keras
import pickle as pkl
import readline
import numpy as np
import pandas as pd

from keras import regularizers
from keras.models import Model, load_model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Activation
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

import keras.backend.tensorflow_backend as K
import tensorflow as tf

#from utils.util import DataManager
#from util import DataManager

import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import gensim
from gensim.parsing.porter import PorterStemmer

import re
from sklearn.model_selection import train_test_split
from keras.utils.generic_utils import get_custom_objects


#路徑
train_path = 'data/training_label.txt'
test_path = 'data/testing_data.txt'
semi_path = 'data/training_nolabel.txt'

"""
#參數
parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('model')
parser.add_argument('action', default='test', choices=['train','test','semi'])

# training argument
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--nb_epoch', default=20, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--gpu_fraction', default=1.0, type=float)
parser.add_argument('--vocab_size', default=None, type=int)
parser.add_argument('--max_length', default=40,type=int)

# model parameter
parser.add_argument('--loss_function', default='binary_crossentropy')
parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU'])
parser.add_argument('-emb_dim', '--embedding_dim', default=300, type=int)
parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('-lr','--learning_rate', default=0.001,type=float)
parser.add_argument('--threshold', default=0.08,type=float)

# output path for your prediction
parser.add_argument('--result_path', default='result.csv',)

# put model in the same directory
parser.add_argument('--load_model', default = None)
parser.add_argument('--save_dir', default = 'model_gensim/')
args = parser.parse_args()
"""
def add_data(name, data_path, with_label=True):
    print ('read data from %s...'%data_path)
    data = {}
    X, Y = [], []
    with open(data_path,'r') as f:
        for line in f:
            if with_label:
                lines = line.strip().split(' +++$+++ ')
                X.append(lines[1])
                Y.append(int(lines[0]))
            else:
                X.append(line)

    if with_label:
        data[name] = [X,Y]
    else:
        data[name] = [X]

    return data

def label_data(data_path):
    data, label = [], []
    
    with open(data_path,'r') as f:
        for line in f:
            line = line.strip().split(' +++$+++ ')
            label.append(int(line[0]))
            data.append(line[1])
    
    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    #word_index = tokenizer.word_index 

    with open('model_t/model256/tokenizer.pkl', 'wb') as f:
        pkl.dump(tokenizer, f)

    data = np.array(pad_sequences(sequences, maxlen=maxlen))
     
    return data, to_categorical(np.array(label), num_classes=2)

def nlabel_data(data_path):
    data = []
    
    with open(data_path,'r') as f:
        for line in f:
            data.append(line.strip())

    return data
 

def test_data(data_path):
    data = []

    with open(data_path,'r') as f:
        next(f)
        for line in f:
            data.append(line.strip().split(',')[1:])

    return data

def tokenize(data, vocab_size):
    print ('create new tokenizer')
    tokenizer = Tokenizer(num_words=None,filters="\n\t")
    for key in data:
        print ('tokenizing %s'%key)
        texts = data[key][0]
        tokenizer.fit_on_texts(texts)

    return tokenizer

def save_tokenizer(tokenizer, path):
    print ('save tokenizer to %s'%path)
    pkl.dump(tokenizer, open(path, 'wb'))

def load_tokenizer(path):
    print ('Load tokenizer from %s'%path)
    tokenizer = pkl.load(open(path, 'rb'))

    return tokenizer

def to_sequence(maxlen, tokenizer, data):
    for key in data:
        print ('Converting %s to sequences'%key)
        tmp = tokenizer.texts_to_sequences(data[key][0])
        data[key][0] = np.array(pad_sequences(tmp, maxlen=maxlen))

    return data

def to_bow(data, tokenizer):
    for key in data:
        print ('Converting %s to tfidf'%key)
        data[key][0] = tokenizer.texts_to_matrix(data[key][0],mode='count')

def to_category(data):
    for key in data:
        if len(data[key]) == 2:
            data[key][1] = np.array(to_categorical(np.asarray(data[key][1]),2),dtype=int)
    return data

def get_data(data, name):
        return data[name]

def split_data(data, name, ratio):
    data = to_category(data)
    data = data[name]
    X = data[0]
    Y = data[1]
    data_size = len(X)
    val_size = int(data_size * ratio)
    return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])

def get_semi_data(data,name,label,threshold,loss_function) : 
        # if th==0.3, will pick label>0.7 and label<0.3
        label = np.squeeze(label)
        index = (label>1-threshold) + (label<threshold)
        semi_X = data[name][0]
        semi_Y = np.greater(label, 0.5).astype(np.int32)
        if loss_function=='binary_crossentropy':
            return semi_X[index,:], semi_Y[index]
        elif loss_function=='categorical_crossentropy':
            return semi_X[index,:], to_categorical(semi_Y[index])
        else :
            raise Exception('Unknown loss function : %s'%loss_function)

def replace_sth(data):
    replace_dict = pkl.load(open('replace_dict.pkl','rb'))
    
    new_data = []
    num = 0
    for str_ in data:
        string = "".join(str_)
        for k,v in replace_dict.items():
            string = string.replace(k,v)
        
        new_data.append(string.strip())

    return new_data

def remove_sth(data):
    new_data = []
    for string in data:
        #移除連續相同的字
        for char in re.findall(r'((\w)\2{2,})', string):
            string = string.replace(char[0], char[1])
        #移除連續重複的標點符號
        for punc in re.findall(r'([-/\\\\()!"+,&?\'.]{2,})',string):
            if punc[0:2] =="..":
                string = string.replace(punc, "...")
            else:
                string = string.replace(punc, punc[0])
        #把所有數字都帶換成相同
        for number in re.findall(r'\d+', string):
            string = string.replace(number, "1")
        new_data.append(string.strip())
   
    return new_data

def stem_sth(data):
    stemmer = gensim.parsing.porter.PorterStemmer()

    new_data = []
    for string in stemmer.stem_documents(data):
        new_data.append(string.strip())
    
    return new_data

def train_word2vec(data, embedding_dim):
    new_data = []
    for key in data:
        for string in data[key][0]:
            new_data.append(string.split(" "))

    model = gensim.models.Word2Vec(new_data, size=embedding_dim, min_count=0, workers=-1)

    return model, len(model.wv.vocab)

def build_matrix(tokenizer, emb_dim, model):
    word_index = tokenizer.word_index
    
    emb_matrix = np.zeros((len(word_index), emb_dim))
    
    for i, v in word_index.items():
        try:
            emb_matrix[v] = model.wv[i]
        except:
            None

    return emb_matrix

def sigmoid_(x):
    return (K.sigmoid(x) * x)
get_custom_objects().update({'sigmoid_': Activation(sigmoid_)})

def simpleRNN(args, word_index, emb_matrix):
    inputs = Input(shape=(args.max_length,)) 
    
    embedding_inputs = Embedding(len(word_index),
                                args.embedding_dim,
                                weights=[emb_matrix],
                                trainable=False)(inputs)

    return_sequence = False
    dropout_rate = args.dropout_rate
    if args.cell == 'GRU':
        RNN_cell = GRU(args.hidden_size, 
                       return_sequences=return_sequence, 
                       dropout=dropout_rate)
    elif args.cell == 'LSTM':
        RNN_cell = Bidirectional(LSTM(256, 
                        activation="tanh",
                        kernel_initializer='he_uniform',
                        return_sequences=True, 
                        dropout=0.2))

    RNN_output1 = RNN_cell(embedding_inputs)
    
    RNN_cell1 = Bidirectional(LSTM(128, 
                        activation="tanh",
                        kernel_initializer='he_uniform',
                        return_sequences=False, 
                        dropout=0.35))
    RNN_output2 = RNN_cell1(RNN_output1)
    #RNN_output3 = BatchNormalization()(RNN_output2)

    outputs = Dense(64, 
                    activation='sigmoid',
                    kernel_regularizer=regularizers.l2(0.0))(RNN_output2)
    outputs = Dropout(0.5)(outputs)

    outputs = Dense(64, 
                    activation='sigmoid',
                    kernel_regularizer=regularizers.l2(0.0))(outputs)
    outputs = Dropout(0.5)(outputs)

    #outputs1 = Dense(1, activation='sigmoid')(outputs)
    outputs2 = Dense(2, activation='softmax')(outputs)
    
    model =  Model(inputs=inputs,outputs=outputs2)

    adam = Adam()
    print ('compile model...')
    
    model.compile( loss='categorical_crossentropy', optimizer=adam, metrics=[ 'acc'])
    
    return model

def simpleRNN_ex():
    inputs = Input(shape=(maxlen,))
    embedding_inputs = Embedding(input_dim=vocab_size, output_dim=emb_dim, trainable=True)(inputs)
    
    
    RNN_cell = LSTM(hid_size, return_sequences=False, dropout=dropout_rate)
    RNN_output = RNN_cell(embedding_inputs)
    outputs = Dense(hid_size//2, activation='relu', kernel_regularizer=regularizers.l2(0.1))(RNN_output)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)
        
    model =  Model(inputs=inputs,outputs=outputs)

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[ 'accuracy',])
    
    return model


def main():
    # limit gpu memory usage
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
    K.set_session(get_session(args.gpu_fraction))
    
    save_path = os.path.join(args.save_dir,args.model)
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir,args.load_model)

    #####read data#####
    #dm = DataManager()
    #print ('Loading data...')
    if args.action == 'train':
        data = add_data('train_data', train_path, True)
    elif args.action == 'semi':
        data = add_data('train_data', train_path, True)
        data = add_data('semi_data', semi_path, False)
    else:
        raise Exception ('Implement your testing parser')
    
    replace_dict = build_replace()
    
    data = stem_sth(remove_sth(replace_sth(data, replace_dict)))
    

    emb_model, voc_size = train_word2vec(data, args.embedding_dim)
    
    # prepare tokenizer
    print ('get Tokenizer...')
    if args.load_model is not None:
        # read exist tokenizer
        tokenizer = load_tokenizer(os.path.join(load_path,'token.pkl'))
    else:
        # create tokenizer on new data
        tokenizer = tokenize(data, args.vocab_size)
        #save_tokenizer(tokenizer, os.path.join(load_path,'token.pkl'))

                            
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path,'token.pkl')):
        save_tokenizer(tokenizer, os.path.join(save_path,'token.pkl')) 

    # convert to sequences
    data = to_sequence(args.max_length, tokenizer, data)
    emb_matrix = build_matrix(tokenizer, args.embedding_dim, emb_model)

    # initial model
    print ('initial model...')
    model = simpleRNN(args, tokenizer.word_index, emb_matrix)    
    print (model.summary())

    if args.load_model is not None:
        if args.action == 'train':
            print ('Warning : load a exist model and keep training')
        path = os.path.join(load_path,'model.h5')
        if os.path.exists(path):
            print ('load model from %s' % path)
            #model.load_weights(path)
            model = load_model('model5/model128/02-0.80380.h5')
        else:
            raise ValueError("Can't find the file %s" %path)
    elif args.action == 'test':
        print ('Warning : testing without loading any model')

     # training
    if args.action == 'train':
        (X,Y),(X_val,Y_val) = split_data(data, 'train_data', args.val_ratio)
        earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')

        #save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path+'{epoch:02d}-{val_acc:.5f}.h5', 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_acc',
                                     mode='max' )
        csv_logger = CSVLogger('train_tt_log.csv', separator=',', append=False)

        history = model.fit(X, Y, 
                            validation_data=(X_val, Y_val),
                            epochs=args.nb_epoch, 
                            batch_size=args.batch_size,
                            callbacks=[checkpoint, earlystopping, csv_logger] )

    # testing
    elif args.action == 'test' :
        raise Exception ('Implement your testing function')

    # semi-supervised training
    elif args.action == 'semi':
        (X,Y),(X_val,Y_val) = split_data(data, 'train_data', args.val_ratio)

        [semi_all_X] = get_data('semi_data')
        earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')

        save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_acc',
                                     mode='max' )
        csv_logger = CSVLogger('semi_tt_log.csv', separator=',', append=True)

        # repeat 10 times
        for i in range(10):
            # label the semi-data
            semi_pred = model.predict(semi_all_X, batch_size=1024, verbose=True)
            semi_X, semi_Y = dm.get_semi_data(data, 'semi_data', semi_pred, args.threshold, args.loss_function)
            semi_X = np.concatenate((semi_X, X))
            semi_Y = np.concatenate((semi_Y, Y))
            print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
            # train
            history = model.fit(semi_X, semi_Y, 
                                validation_data=(X_val, Y_val),
                                epochs=2, 
                                batch_size=args.batch_size,
                                callbacks=[checkpoint, earlystopping, csv_logger] )

            if os.path.exists(save_path):
                print ('load model from %s' % save_path)
                model.load_weights(save_path)
            else:
                raise ValueError("Can't find the file %s" %path)

#if __name__ == '__main__':
#        main()
maxlen = 39
vocab_size = 25000
emb_dim = 128
cell = 'LSTM'
hid_size = 512
dropout_rate = 0.3

#load_path = './model_gensim/model128/'
model_path = sys.argv[5]#'mymodel.h5'
token_path = sys.argv[4]#'token_new.pkl'

test_path = sys.argv[1]#'testing_data.txt'
output_path = sys.argv[2]#'sample.csv'
mode = sys.argv[3]
#model = simpleRNN()    
#model.load_weights(load_path+model_path)
model = load_model(model_path)
print('load model ok')

print(model.summary())

data = test_data(test_path)
print(len(data))
print('read test_data ok')

data = stem_sth(remove_sth(replace_sth(data)))
print(len(data))
print('process test_data ok')

tokenizer = pkl.load(open(token_path,'rb'))
sequences = tokenizer.texts_to_sequences(data)
data = np.array(pad_sequences(sequences, maxlen=maxlen))
print('tokenizer ok')
print(data.shape)
#pred = np.around(model.predict(data, batch_size=1024, verbose=True))
pred = np.argmax(model.predict(data,verbose=True),axis=1)
#result = np.argmax(pred_y,axis=1)
print('pred ok')

with open(output_path, 'w') as f:
    f.write('id,label\n')

    for i,v in enumerate(pred):
        f.write('{},{}\n'.format(i,int(v)))

print('output ok')
