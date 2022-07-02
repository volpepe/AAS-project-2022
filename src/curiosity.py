from typing import Dict, Tuple
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, Model
from tensorflow.keras.layers import Layer

class ICM(Model):
    '''
    Implementation of the ICM (Intrinsic Curiosity Module).

    This module computes curiosity, an intrinsic reward for the agent that is
    added to the extrinsic reward it receives at each time step, but
    may not receive for a long sequence of steps due to the sparsity of
    rewards.
    An agent that is curious is able to navigate the environment 
    effectively even though no explicit rewards are given to it.

    The intrinsic reward can be defined as the difference between what
    the agent expected its action to do and what its action actually does.
    If the difference is high, it means that the new state of the environment
    after the action was highly unexpected by the agent, so it's an incentive
    to explore further.

    In practice, a neural network tries to predict the next state St+1 after action
    At from state St and the curiosity intrinsic reward is given by the difference between
    the predicted St+1 and the actual St+1. However we don't predict full states: instead
    we work in a low dimensional space that should only encode the meaningful parts
    of the states, making this system more resistent to noise and small variations.

    To learn a meaningful state representation, we let an inverse model predict the action
    that was made by the agent given state St and state St+1. The low-dimensional space
    learnt by this inverse model is shared with the "forward" model which is the one computing
    state St+1 given the encoding of statr St and the action At.
    '''
    def __init__(self, num_actions, optimizer=optimizers.Adam(learning_rate=10**-3), 
                 beta=0.1, eta=0.01, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_actions = num_actions
        self.beta = beta                # Weight of the forward model loss against the inverse model loss
        self.eta = eta                  # Scaling factor for the intrinsic reward signal
        self.optimizer = optimizer      # The optimizer to be used in training
        self.statistics = self.new_stats_dict()         # To be used in training to collect aggregated statistics about an episode
        self.encoding_layer = EncodingLayer()
        self.forward_model = ForwardModel(num_actions)
        self.inverse_model = InverseModel(num_actions)

    def new_stats_dict(self) -> Dict:
        return {
            'loss_inverse': [],
            'loss_forward': [],
            'total_loss': []
        }

    def end_episode(self) -> Dict:
        '''
        Returns the dictionary of statistics for this episode and clears it for the 
        next episode.
        '''
        stats = {k: self.statistics[k] for k in self.statistics}
        self.statistics = self.new_stats_dict()
        return stats

    def change_scaling_factor(self, new_eta) -> None:
        '''
        Function that allows dynamic changes to the scaling factor.
        '''
        self.eta = new_eta        

    def compute_values(self, inputs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        st, at, st1 = inputs
        # Computing state encodings
        e_st, e_st1 = self.encoding_layer((st, st1))
        # Predict the encoding of state st1 and the action.
        pred_e_st1 = self.forward_model((at, e_st))
        pred_at = self.inverse_model((e_st, e_st1))
        return e_st, e_st1, pred_e_st1, pred_at

    def call(self, inputs, training=False) -> tf.Tensor:
        '''
        Note: when training=True we are also computing the loss and a training step
        for the model.
        '''
        # Inputs are the state St, action At and state St+1
        # States are [1,42,42,4] tensors, while action At is a [1,num_actions] tensor
        st, at, st1 = inputs
        if training:
            with tf.GradientTape() as tape:
                e_st, e_st1, pred_e_st1, pred_at = self.compute_values(inputs)
                # We compute the loss of the ICM. It's a composite loss, because we have two 
                # communicating modules:
                # - The loss of the forward model is a regression loss between the 
                #   ground truth encoding and the predicted one
                # - The loss of the inverse model is a cross-entropy loss between the
                #   ground truth action probability distribution and the predicted one.
                loss_inverse = losses.categorical_crossentropy(at, pred_at)
                loss_forward = losses.mean_squared_error(e_st1, pred_e_st1)
                loss_value = (1-self.beta)*tf.reduce_sum(loss_inverse) + self.beta*tf.reduce_sum(loss_forward)
            # Compute the gradient and apply it
            gradients = tape.gradient(loss_value, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            # Update statistics
            self.statistics['loss_inverse'].append(loss_inverse)
            self.statistics['loss_forward'].append(loss_forward)
            self.statistics['total_loss'].append(loss_value)
        else:
            # Simply obtain the values calling the networks. Do not compute the loss.
            e_st, e_st1, pred_e_st1, pred_at = self.compute_values(inputs)
        # Finally, compute the output (intrinsic reward)
        ri = self.eta/2*tf.norm(pred_e_st1 - e_st1)
        return ri


class EncodingLayer(Layer):
    '''
    Utility layer for computing the encodings of the states, separated from the rest since 
    encodings are shared between the inverse and forward models.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.conv1   = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu')
        self.conv2   = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu')
        self.conv3   = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu')
        self.conv4   = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='elu')
        self.flatten = layers.Flatten()
    
    def call(self, inputs) -> Tuple[tf.Tensor, tf.Tensor]:
        # Inputs are the states St and St+1, [1, 42, 42, 4] tensors
        st, st1 = inputs
        # Compute encoding of state st and st1
        # 1x288 <- 1x3x3x32 <- 1x6x6x32 <- 1x11x11x32 <- 1x21x21x32 <- 1x42x42x4
        e_st  = self.flatten(self.conv4(self.conv3(self.conv2(self.conv1(st)))))
        e_st1 = self.flatten(self.conv4(self.conv3(self.conv2(self.conv1(st1)))))
        return e_st, e_st1


class ForwardModel(Layer):
    '''
    The forward model of the ICM takes as input the action At (one-hot encoded)
    and the encoding of the state St (`e(St)`). It tries to predict the encoding
    of state St+1 (`pred[e(St+1)]`)
    '''
    def __init__(self, num_actions, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_actions = num_actions
        self.concat = layers.Concatenate(axis=1)
        self.dense1 = layers.Dense(256, activation='elu')
        self.dense2 = layers.Dense(288, activation='elu')

    def call(self, inputs) -> tf.Tensor:
        # Inputs: the action At and the encoding of the state e(St)
        at, e_st = inputs
        # at is [1, num_actions]
        # enc_st is [1, 288]
        x = self.concat([at, e_st])                 # [1, num_actions + 288]
        pred_e_st1 = self.dense2(self.dense1(x))    # [1, 288]
        return pred_e_st1


class InverseModel(Layer):
    '''
    The inverse model of the ICM takes as input the encoding of states St and St+1
    `e(St)` and `e(St+1)` and tries to predict the action (`pred[At]`).
    '''
    def __init__(self, num_actions, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_actions = num_actions
        self.concat  = layers.Concatenate(axis=1)
        self.dense1  = layers.Dense(256, activation='relu')
        self.dense2  = layers.Dense(self.num_actions, activation='softmax')

    def call(self, inputs) -> tf.Tensor:
        e_st, e_st1 = inputs
        # Concatenate the encodings
        e_states = self.concat([e_st, e_st1])             # [1, 288*2]
        # Dense layers for action prediction
        pred_at = self.dense2(self.dense1(e_states))      # [1, num_actions], probability distribution
        return pred_at
