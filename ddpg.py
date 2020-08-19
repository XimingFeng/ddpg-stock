import gym
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from replay_buffer import Buffer
import pandas as pd


class ActorCritic:

    def __init__(self, state_dim, action_dim, low_bound_act=-2, high_bound_act=2, gamma=0.99,
                 ,actor_hp_param=dict(), critic_hp_param=dict()):
        self.gamma = tf.convert_to_tensor(gamma)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.main_actor = self.build_actor()
        self.main_critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        self.main_actor.set_weights(self.target_actor.get_weights())
        self.main_critic.set_weights(self.target_critic.get_weights())
        self.low_bound_act = low_bound_act
        self.high_bound_act = high_bound_act
        print("Actor Network Summary: ")
        self.main_actor.summary()
        print("Critic Network Summary: ")
        self.main_critic.summary()

    def train(self, env, num_eps, actor_lr, critic_lr, train_every_step=1, batch_size=1):
        replay_buffer = Buffer(self.state_dim, self.action_dim, batch_size=batch_size)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        rewards = []
        for eps in range(num_eps):
            current_state = env.reset()
            step_count = 0
            done = False
            episodic_reward = 0
            while done is False:
                step_count += 1
                current_state_tf = tf.expand_dims(tf.convert_to_tensor(np.squeeze(current_state)), axis=0)
                actor_out = self.main_actor(current_state_tf)
                action = self.get_action(actor_out)
                nxt_state, reward, done, info = env.step(action)
                episodic_reward += reward
                replay_buffer.insert(current_state, action, reward, nxt_state, done)
                if step_count % train_every_step == 0:
                    train_batch = replay_buffer.sample_batch()
                    self.one_step_train(train_batch, actor_optimizer, critic_optimizer)
                current_state = nxt_state
            replay_buffer.reset_buffer()
            rewards.append(episodic_reward)
            print(f"Episode {eps}, reward: {episodic_reward}")
        return rewards

    def test(self, env):
        current_state = env.reset()
        done = False
        while done is False:
            # print(f"-------------- step {step_count} --------------------")
            current_state_tf = tf.expand_dims(tf.convert_to_tensor(np.squeeze(current_state)), axis=0)
            actor_out = self.main_actor(current_state_tf)
            action = self.get_action(actor_out)
            nxt_state, reward, done, info = env.step(action)
            current_state = nxt_state
            env.render()
        env.close()

    def get_action(self, actor_output):
        return np.clip(actor_output, self.low_bound_act, self.high_bound_act)

    def one_step_train(self, batch_replay, actor_optimizer, critic_optimizer):
        states = tf.convert_to_tensor(batch_replay[0])
        actions = tf.convert_to_tensor(batch_replay[1])
        rewards = tf.cast(tf.convert_to_tensor(batch_replay[2]), tf.float32)
        next_states = tf.convert_to_tensor(batch_replay[3])
        is_term = tf.cast(tf.convert_to_tensor(batch_replay[4]), tf.float32)
        # print(f'State: {states}')
        # print(f"Action: {actions}")
        # print(f"Rewards:  {rewards}")
        # print(f"Nest States: {next_states}")
        # print(f"Is terminal states: {is_term}")
        # Update the parameters of main critic to minimize Mean Squared Bellman Error
        with tf.GradientTape() as tape:
            target_nxt_actions = self.target_actor(next_states)
            target_nxt_actions = self.get_action(target_nxt_actions)
            target_nxt_value = self.target_critic([next_states, target_nxt_actions])
            # print(f"target critic: {target_nxt_value}")
            target_value = rewards + self.gamma * (1 - is_term) * target_nxt_value
            main_value = self.main_critic([states, actions])
            critic_loss = tf.math.reduce_mean(tf.math.square(main_value - target_value))
        critic_gradient = tape.gradient(critic_loss, self.main_critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_gradient, self.main_critic.trainable_variables))

        # Update the parameters of main actor to maximize action-value function
        with tf.GradientTape() as tape:
            main_action = self.main_actor(states)
            main_value = -tf.math.reduce_mean(self.main_critic([states, main_action]))
        actor_gradient = tape.gradient(main_value, self.main_actor.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_gradient, self.main_actor.trainable_variables))

        # Update target network weights
        self.update_target_weights()

    def update_target_weights(self, rho=0.995):
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
        inputs = layers.Input(shape=self.state_dim)
        out = layers.Dense(512, activation="relu")(inputs)
        out = layers.Dense(self.action_dim[0], activation=None)(out)
        outputs = layers.GaussianNoise(stddev=0.2)(out)
        return Model(inputs, outputs)

    def build_critic(self):
        # state as input
        state_input = layers.Input(shape=self.state_dim)
        state_out = layers.Dense(16, activation="relu")(state_input)

        # action as input
        action_input = layers.Input(shape=self.action_dim)
        action_out = layers.Dense(32, activation="relu")(action_input)
        state_act_concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(512, activation="relu")(state_act_concat)

        outputs = layers.Dense(1)(out)
        return Model([state_input, action_input], outputs)


