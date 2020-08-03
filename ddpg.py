import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from replay_buffer import Buffer
import pandas as pd


class ActorCritic:

    def __init__(self, actor_hp_param=dict(), critic_hp_param=dict(), gamma=0.99):
        self.main_actor = self.build_actor()
        self.main_critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        self.gamma = gamma

    def train(self, env, num_eps, actor_lr, train_every_step, critic_lr):
        replay_buffer = Buffer()
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        for _ in range(num_eps):
            current_state = env.reset()
            step_count = 0
            while self.is_term_state(current_state) is False:
                step_count += 1
                action = self.main_actor(current_state)
                nxt_state, reward, done, info = env.step(action)
                replay_buffer.insert(current_state, action, reward, nxt_state, done)
                if step_count == train_every_step:
                    train_batch = replay_buffer.sample_batch()
                    self.one_step_train(train_batch, actor_optimizer, critic_optimizer)
                current_state = nxt_state

    def one_step_train(self, batch_replay, actor_optimizer, critic_optimizer):
        states = tf.convert_to_tensor(batch_replay[0])
        actions = tf.convert_to_tensor(batch_replay[1])
        rewards = tf.convert_to_tensor(batch_replay[2])
        next_states = tf.convert_to_tensor(batch_replay[3])
        is_term = tf.convert_to_tensor(batch_replay[4])

        # Update the parameters of main critic to minimize Mean Squared Bellman Error
        with tf.GradientTape() as tape:
            target_nxt_actions = self.target_actor(next_states)
            target_nxt_value = self.target_critic([next_states, target_nxt_actions])
            target_value = rewards + self.gamma * (1 - is_term) * target_nxt_value
            main_value = self.main_critic([states, actions])
            critic_loss = tf.math.reduce_mean(tf.math.square(main_value - target_value))
        critic_gradient = tape.gradient(critic_loss, self.main_critic.trainable_variables)
        critic_optimizer.apply_gradient(zip(critic_gradient, self.main_critic.trainable_variables))

        # Update the parameters of main actor to maximize action-value function
        with tf.GradientTape() as tape:
            main_action = self.main_actor(states)
            main_value = -tf.math.reduce_mean(self.main_critic([states, main_action]))
        actor_gradient = tape.gradient(main_value, self.main_actor.trainable_variables)
        actor_optimizer.apply_gradient(zip(actor_gradient, self.main_actor.trainable_vaables))

        # Update target network weights
        self.update_target_weights()

    def update_target_weights(self, rho=0.95):
        # Update target critic weights a little bit towards main critic's weights
        new_weights = []
        target_critic_weights = self.target_critic.weights
        for i, weight in enumerate(self.main_critic.weights):
            new_weights.append(rho * target_critic_weights[i] + (1 - rho) * weight)
        self.target_critic.set_weights(new_weights)

        # Update target actor weights a little bit towards main actor's weights
        new_weights = []
        target_actor_weights = self.target_actor.weights
        for i, weight in enumerate(self.main_actor.weights):
            new_weights.append(rho * target_actor_weights[i] + (1 - rho) * weight)
        self.target_actor.set_weights(new_weights)

    def build_actor(self):
        inputs = layers.Input(shape=())
        out = layers.Dense(512, activation="relu")(inputs)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(512, activation="relu")(out)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(1, activation=None)(out)
        noise = tf.random.normal(shape=tf.shape(out), stddev=1)
        outputs = out + noise
        return Model(inputs, outputs)

    def build_critic(self):
        # state as input
        state_input = layers.Input(shape=())
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.BatchNormalization()(state_out)
        state_out = layers.Dense(32, activation="relu")(state_out)
        state_out = layers.BatchNormalization()(state_out)

        # action as input
        action_input = layers.Input(shape=())
        action_out = layers.Dense(32, activation="relu")(action_input)
        action_out = layers.BatchNormalization()(action_out)
        state_act_concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(512, activation="relu")(state_act_concat)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(512, activation="relu")(out)
        out = layers.BatchNormalization()(out)
        outputs = layers.Dense(1)(out)
        return Model([state_input, action_input], outputs)


