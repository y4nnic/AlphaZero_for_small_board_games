import tensorflow as tf
import config

from time import time

from keras.models import Model, model_from_json
from keras.optimizers import SGD#Adam
from keras import regularizers
from keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, Flatten, Input
from keras.callbacks import TensorBoard


class NeuralNetwork:

    def __init__(self, input_shape, num_possible_moves, load_path=None, net_id="test", lr=None, reg=None):
        """ TODO docstring __init__"""
        self.input_shape = input_shape
        self.tensorboard = None
        self.id = net_id

        self.lr = lr
        self.reg = reg

        if load_path is None:
            self.network = self._compile(
                reg_strength=config.NEURAL_NETWORKS['regularization_strength'],
                num_residual_blocks=config.NEURAL_NETWORKS['num_residual_blocks'],
                num_filters=config.NEURAL_NETWORKS['num_filters_tower'], #32,
                kernel_size=config.NEURAL_NETWORKS['kernel_size_tower'], #3,
                momentum=0.99,
                epsilon=0.001,
                hidden_dim=config.NEURAL_NETWORKS['hidden_dim_value'], #64,
                num_possible_moves=num_possible_moves
        )
        else:
            file = open(load_path, "r")
            model_json = file.read()
            file.close()

            self.network = model_from_json(model_json)
            load_path = load_path[:-4] + "h5"
            self.network.load_weights(load_path)

            self.network.compile(
                optimizer = SGD(
                    lr=config.NEURAL_NETWORKS['learning_rate'],
                    momentum=0.9
                ),
                #optimizer=Adam(
                #    lr=config.NEURAL_NETWORKS['learning_rate'],
                #),
                loss={
                    'value': 'mean_squared_error',
                    'policy': cross_entropy
                },
                loss_weights={
                    'value': 0.5,
                    'policy': 0.5
                }
            )

    def predict(self, state):
        """ TODO docstring predict """
        return self.network.predict_on_batch(state)

    def fit(self, X, y, batch_size, epochs, verbose):
        """ TODO docstring fit """
        self.network.fit(
            x=X,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=0.2,
            callbacks=[self.tensorboard]
        )

    def policy_head(self, X, num_possible_moves, reg, momentum=0.99, epsilon=0.001):
        """ TODO docstring policy head """
        conv_block = self.convolutional_block(
            X,
            num_filters=config.NEURAL_NETWORKS['num_filters_policy'],
            kernel_size=1,
            momentum=momentum,
            epsilon=epsilon,
            reg_strength=reg
        )

        flatten = Flatten()(conv_block)

        policy = Dense(
            num_possible_moves,
            name='policy',
            kernel_regularizer=regularizers.l2(reg)
        )(flatten)

        tf.summary.histogram("policy", policy)

        return policy

    def value_head(self, X, reg, hidden_dimension=256, momentum=0.99, epsilon=0.001):
        """ TODO docstring value_head """
        conv_block = self.convolutional_block(
            X,
            num_filters=config.NEURAL_NETWORKS['num_filters_value'],
            kernel_size=1,
            momentum=momentum,
            epsilon=epsilon,
            reg_strength=reg
        )

        flatten = Flatten()(conv_block)

        dense_1 = Dense(
            hidden_dimension,
            kernel_regularizer=regularizers.l2(reg)
        )(flatten)

        relu = Activation('relu')(dense_1)

        dense_2 = Dense(
            1,
            kernel_regularizer=regularizers.l2(reg)
        )(relu)

        value = Activation('tanh', name='value')(dense_2)

        tf.summary.histogram("value", value)

        return value

    def convolutional_block(self, X, num_filters, kernel_size, reg_strength, momentum=0.99, epsilon=0.001):
        """ TODO docstring convolutional_block """
        conv = Conv2D(
            num_filters,
            kernel_size,
            data_format="channels_last",
            strides=(1, 1),
            padding='same',
            kernel_regularizer=regularizers.l2(reg_strength)
        )(X)

        batch_norm = BatchNormalization(
            axis=3,
            momentum=momentum,
            epsilon=epsilon
        )(conv)

        output = Activation('relu')(batch_norm)

        return output

    def residual_block(self, X, num_filters, kernel_size, reg_strength, momentum=0.99, epsilon=0.001):
        """ TODO docstring residual_block """
        conv_block = self.convolutional_block(X, num_filters, kernel_size, reg_strength, momentum, epsilon)

        conv = Conv2D(
            num_filters,
            kernel_size,
            data_format='channels_last',
            strides=(1,1),
            padding='same',
            kernel_regularizer=regularizers.l2(reg_strength)
        ) (conv_block)

        batch_norm = BatchNormalization(
            axis=3,
            momentum=momentum,
            epsilon=epsilon
        )(conv)

        skip_connection = Add()([X, batch_norm])

        output = Activation('relu')(skip_connection)

        return output

    def _compile(self, reg_strength, num_residual_blocks ,num_filters, kernel_size, momentum, epsilon, hidden_dim, num_possible_moves):
        """ TODO docstring _compile """
        input = Input(shape=self.input_shape)

        conv_block = self.convolutional_block(
            input,
            num_filters,
            kernel_size,
            reg_strength,
            momentum,
            epsilon
        )

        residual_tower = conv_block
        for i in range(num_residual_blocks):
            residual_tower = self.residual_block(
                residual_tower,
                num_filters,
                kernel_size,
                reg_strength,
                momentum,
                epsilon
            )

        value_head = self.value_head(
            residual_tower,
            hidden_dimension=hidden_dim,
            reg=reg_strength,
            momentum=momentum,
            epsilon=epsilon
        )

        policy_head = self.policy_head(
            residual_tower,
            num_possible_moves=num_possible_moves,
            reg=reg_strength,
            momentum=momentum,
            epsilon=epsilon
        )

        model = Model(inputs=[input], outputs=[value_head, policy_head])

        model.compile(
            optimizer = SGD(
                lr = config.NEURAL_NETWORKS['learning_rate'],
                momentum = 0.0
            ),
            #optimizer=Adam(
            #    lr= config.NEURAL_NETWORKS['learning_rate']
            #),
            loss={
                'value': 'mean_squared_error',
                'policy': cross_entropy
            },
            loss_weights={
                'value': 0.5,
                'policy': 0.5
            }
        )

        self.tensorboard = TensorBoard(log_dir="tb/{}".format(self.id))

        return model

    def to_json(self):
        """ TODO docstring to_json """
        return self.network.to_json()

    def save_weights(self, path):
        """ TODO docstring save_weights"""
        self.network.save_weights(path)


def cross_entropy(y_true, y_pred):
    """ TODO docstring cross_entropy """
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_true,
        logits=y_pred
    )

    return loss

