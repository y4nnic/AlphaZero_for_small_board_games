import tensorflow as tf
import config

from time import time

from keras.models import Model, model_from_json
from keras.optimizers import SGD#Adam
from keras import regularizers
from keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, Flatten, Input
from keras.callbacks import TensorBoard


class NeuralNetwork:
    """ The neural network evaluates states for the MCTS and is trained on the positions
    saved in the position memory. It is controled by an AZModel object.

    Attributes:
        input_shape: Shape of the neural network's input: (board_height, board_width, 3).
        tensorboard: Instance of the TensorBoard class. It is used to save and visualize the loss.
        id: Unique name or number of this neural network version.
        lr: The inital learning rate that is used to train the network. If None, the config.py
            file will provide a value.
        reg: The regularization strength used to train the network. If None, the config.py
            file will provide a value.
        network: The instance of the Keras Model class representing the neural network.
    """

    def __init__(self, input_shape, num_possible_moves, load_path=None, net_id="test", lr=None, reg=None):
        """ Iniitalizes the neural network if loadpath is None. Otherwise it is loaded from
        the given files.

        Args:
            input_shape: Shape of the neural network's input: (board_height, board_width, 3).
            num_possible_moves: Maximum number of possible moves for one state of the
                considered game.
            load_path: Path of the saved network's file that will be loaded.
            net_id: Unique name or number of this neural network version.
            lr: The inital learning rate that is used to train the network. If None, the config.py
                file will provide a value.
            reg: The regularization strength used to train the network. If None, the config.py
                file will provide a value.
        """
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
        """ The neural networks predicts a value and a policy for a given state.

        Args:
             state: Tensor of shape (board_height, board_width, 3) corresponding to
                the current position.
        """
        return self.network.predict_on_batch(state)

    def fit(self, X, y, batch_size, epochs, verbose):
        """ The network is trained on the data (X,y).

        Args:
             X: The training data.
             y: The targets (for value and policy head).
             batch_size: The batch-size used during the training.
             epochs: The number of epochs used during the training.
        """
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
        """ The policy head gets the residual tower's output as input and outputs
         a policy estimate.

        Args:
             X: Input.
             num_possible_moves: Maximum number of possible moves for one state of the
                considered game.
             reg: The regularization strength used during training.
        """
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
        """ The value head gets the residual tower's output as input and outputs a
         value estimate.

        Args:
            X: Input.
            reg: The regularization strength used during training.
            hidden_dimension: Output dimension of the first fully-connected layer.
        """
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
        """ convolution -> BatchNorm -> ReLU

        Args:
            X: The input.
            num_filters: The number of filters for the convolution.
            kernel_size: The kernel-size for the convolution.
            reg_strength: The regularization strength used during training.
        """
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
        """ conv. block -> convolution -> BatchNorm -> skip connection -> ReLU

        Args:
            X: The input.
            num_filters: The number of filters for the convolution.
            kernel_size: The kernel-size for the convolution.
            reg_strength: The regularization strength used during training.
        """
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
        """ This method compiles the full neural network's graph.

        Args:
             reg_strength: The regularization strength used during training.
             num_residual_blocks: The number of residual blocks used in the architecture.
             num_filters: The number of filters used in the convolutional layers.
             kernel_size The kernel-size used in the convolutional layers.
             momentum: Momentum of the batch normalization modules.
             epsilon: Epsilon of the batch normalization modules.
             hidden_dim: The output dimension of the first fully connected layer in the value head.
             num_possible_moves: Maximum number of possible moves for one state of the
                considered game.num_possible_moves:
        """
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
        """ Saves the current neural network to a .json file. """
        return self.network.to_json()

    def save_weights(self, path):
        """ Saves the weights of the current neural network to a .h5 file.

        Args:
            path: Path of the directory in which the weights will be saved.
        """
        self.network.save_weights(path)


def cross_entropy(y_true, y_pred):
    """ The cross entropy loss function for the policy head.

    Args:
        y_true: Targets.
        y_pred: Predictions of the neural network.
    """
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_true,
        logits=y_pred
    )

    return loss

