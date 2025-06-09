from keras.models import Model
from keras.layers import Input, Dense, Embedding, Flatten, Dropout, Activation, add
from keras.regularizers import l2

def create(I, U, K, hidden_activation, output_activation, q=0.5, l=0.01):
    '''
    Create CDAE model.
    Reference:
      Yao Wu, Christopher DuBois, Alice X. Zheng, Martin Ester.
      Collaborative Denoising Auto-Encoders for Top-N Recommender Systems.
      WSDM 2016.

    :param I: number of items
    :param U: number of users
    :param K: number of units in hidden layer
    :param hidden_activation: activation function of hidden layer
    :param output_activation: activation function of output layer
    :param q: drop probability
    :param l: L2 regularization parameter
    :return: Keras model
    '''
    x_item = Input(shape=(I,), name='x_item')
    h_item = Dropout(q)(x_item)
    h_item = Dense(K, kernel_regularizer=l2(l), bias_regularizer=l2(l))(h_item)

    x_user = Input(shape=(1,), dtype='int32', name='x_user')
    h_user = Embedding(input_dim=U, output_dim=K, embeddings_regularizer=l2(l))(x_user)
    h_user = Flatten()(h_user)

    h = add([h_item, h_user])

    if hidden_activation:
        h = Activation(hidden_activation)(h)

    y = Dense(I, activation=output_activation)(h)

    return Model(inputs=[x_item, x_user], outputs=y)
