from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np

import sys 

def autoencoder_(X,x_train,x_val):
    # build model
    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    # build encoder
    encoder = Model(input=input_img, output=encoded)

    # build autoencoder
    adam = Adam(lr=5e-4)
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.summary()

    # train autoencoder
    autoencoder.fit(x_train, x_train,
                    epochs=120,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_val, x_val))
    autoencoder.save('autoencoder120.h5')
    encoder.save('encoder120.h5')

    encoder = load_model('encoder120.h5')

    # after training, use encoder to encode image, and feed it into Kmeans
    encoded_imgs = encoder.predict(X)
    encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)

    return encoded_imgs

def pca_(n_com,X):
    encoder = PCA(n_components=n_com, copy=True, whiten=True, svd_solver='full', tol=0.0, iterated_power='auto', random_state=0)
    #encoder.save('pca400.h5')

    #encoder = load_model('pca400.h5')
    encoded_imgs = encoder.fit_transform(X)
    encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)

    return encoded_imgs

# load images
train_num = 130000
X = np.load(sys.argv[1])
X = X.astype('float32') / 255.
X = np.reshape(X, (len(X), -1))
x_train = X[:train_num]
x_val = X[train_num:]
#x_train.shape, x_val.shape

n_com = 400
use_pca = True


encoded_imgs = pca_(n_com,X)
kmeans = KMeans(n_clusters=2, random_state=0, n_jobs=-1, n_init=10, max_iter=300).fit(encoded_imgs)


# get test cases
f = pd.read_csv(sys.argv[2])
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])

# predict
o = open(sys.argv[3], 'w')
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1
    else: 
        pred = 0
    o.write("{},{}\n".format(idx, pred))
o.close()