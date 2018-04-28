"""Agent class to handle TA and withdrawal actions."""
import logging
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Embedding, LSTM, Activation, Dropout
from keras.optimizers import Adam

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import numpy as np
import random
from collections import deque
import itertools

logging.basicConfig(filename='./qatrader.log')


class AgentTa(object):
    """TA agent class."""

    ACTION_SIT = 0
    ACTION_BUY = 1
    ACTION_SELL = 2
    ACTION_WITHDRAW1 = 3
    ACTION_REFUND1 = 4

    def __init__(
        self,
        state_size,
        is_eval=False,
        neural_network='rnn',
        model_path="."
    ):
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # sit, buy, sell, withdraw 10% , or 25%
        self.memory = deque(maxlen=3000)
        self.neural_network = neural_network
        self.model_name = model_path
        self.is_eval = is_eval

        self.fit_epochs = 1
        self.gamma = 0.75
        # self.gamma = 0.95
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.003

        print(
            "Loading model with state size {}: {}".format(
                self.state_size,
                model_path
            )
        )

        self.load(
            model_path,
            is_eval=is_eval
        )

    def load(self, filename=None, is_eval=False):
        """Instantiate the model or load its file."""
        self.model = load_model(filename) \
            if is_eval is True \
            else self._model()

    def _model(self):
        return getattr(
            self,
            "{}_model".format(self.neural_network)
        )()

    def rnn_model(self):
        """Return a RNN-based (q-learning) model."""
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.05
        set_session(tf.Session(config=config))

        model = Sequential()
        model.add(
            Dense(
                units=64,
                input_dim=self.state_size,
                activation="relu"
            )
        )
        model.add(
            Dense(
                units=32,
                activation="relu"
            )
        )
        model.add(
            Dense(
                units=8,
                activation="relu"
            )
        )
        model.add(
            Dense(
                units=self.action_size,
                activation="linear"
            )
        )
        model.compile(
            loss="mse",
            optimizer=Adam(lr=self.learning_rate)
        )
        return model

    def lstm_model(self):
        """Return a LSTM-based model."""
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.15
        set_session(tf.Session(config=config))

        model = Sequential()
        model.add(
            Embedding(
                self.state_size,
                64
            )
        )
        model.add(
            LSTM(
                64,
                return_sequences=True,
                input_shape=(
                    1,
                    self.state_size
                ),
                activation='sigmoid'
            )
        )
        model.add(Dropout(0.25))
        # model.add(
        #     LSTM(
        #         64,
        #         return_sequences=True,
        #         activation='sigmoid'
        #     )
        # )
        # model.add(
        #     Dropout(0.25)
        # )
        # model.add(
        #     LSTM(
        #         64,
        #         activation='sigmoid'
        #     )
        # )
        # model.add(Dropout(0.25))
        model.add(
            Dense(
                units=self.action_size,
                activation='sigmoid'
            )
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=0.003),
            metrics=['accuracy', 'categorical_accuracy']
        )
        print(model.summary())
        # model.add(
        #     Embedding(
        #         self.state_size,
        #         64
        #     )
        # )
        # model.add(
        #     LSTM(
        #         64,
        #         input_dim=self.state_size,
        #         dropout=0.2,
        #         recurrent_dropout=0.2
        #     )
        # )
        # model.add(
        #     Dense(
        #         units=self.action_size,
        #         activation='sigmoid'
        #     )
        # )

        # model.compile(
        #     loss='binary_crossentropy',
        #     optimizer=Adam(lr=0.003),
        #     metrics=['accuracy', 'categorical_accuracy']
        # )
        return model

    def act(self, state):
        """Take an action based on a given state.

        Given an array of states for the current situation, the
        system chooses one action over all the available ones.available

        This decisions can either be taken randomly or based on a model
        prediction, depending on the value of epsilon. This value decays
        over every experiment_replay call, at a epsilon_decay ratio,
        which means that the more experiment_replay calls, the more
        prediction the model is *willing* to make.
        """
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return (
                " ",
                random.randrange(self.action_size),
                [1 / self.action_size] * self.action_size
            )

        options = self.model.predict(state)

        return (
            "*",
            np.argmax(options[0]),
            options[0]
        )

    def experiment_replay(self, batch_size):
        """Replay the experiment in to get feedback.

        Takes the last *batch_size* experiments and runs them through
        a model.predict -> model.fit based on the last rewards
        """
        mini_batch = []
        # mem_len = len(self.memory)

        # mini_batch = random.sample(
        #     itertools.islice(self.memory, mem_len - batch_size, mem_len)
        # )
        mini_batch = random.sample(
            self.memory,
            batch_size
        )

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            # print(self.model.predict(next_state))
            if not done:
                target += self.gamma * \
                    np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=self.fit_epochs, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
