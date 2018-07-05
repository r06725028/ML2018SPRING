from keras.models import Model
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Activation,
    Bidirectional,
    Embedding,
    PReLU,
    GRU,
    Input,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    concatenate,
    Dot,
)


def gen_model(weights=None):
    vocab_size, data_dim = weights.shape

    # Question model
    q_input = Input(shape=(71,))
    q_embedding = Embedding(vocab_size, data_dim, weights=[weights], trainable=False)(q_input)
    q_gru_1 = Bidirectional(GRU(80, return_sequences=True))(q_embedding)
    q_gru_2 = Bidirectional(GRU(80, return_sequences=True))(q_gru_1)
    q_avg_pool = GlobalAveragePooling1D()(q_gru_2)
    q_max_pool = GlobalMaxPooling1D()(q_gru_2)
    q_concat = concatenate([q_max_pool, q_avg_pool])
    q_dense = Dense(40)(q_concat)

    # Answer model
    a_input = Input(shape=(44,))
    a_embedding = Embedding(vocab_size, data_dim, weights=[weights], trainable=False)(a_input)
    a_gru_1 = Bidirectional(GRU(80, return_sequences=True))(a_embedding)
    a_gru_2 = Bidirectional(GRU(80, return_sequences=True))(a_gru_1)
    a_avg_pool = GlobalAveragePooling1D()(a_gru_2)
    a_max_pool = GlobalMaxPooling1D()(a_gru_2)
    a_concat = concatenate([a_max_pool, a_avg_pool])
    a_dense = Dense(40)(a_concat)

    # Interact layer
    output = Activation('sigmoid')(Dot(axes=1)([q_dense, a_dense]))

    model = Model(inputs=[q_input, a_input], outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
