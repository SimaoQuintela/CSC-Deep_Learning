import gym
import pickle
from pprint import pprint
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import layers  


class PlayerAgent:
    
    def __init__(self, game_steps, nr_of_games_for_training, score_requirement, game="CartPole_v1", log=False):
        self.game_steps = game_steps
        self.nr_of_games_for_training = nr_of_games_for_training
        self.score_requirement = score_requirement
        self.game = game
        self.log = log
        self.scores = []

        
    def play_random_games(self):
        if self.log:
            env = gym.make('CartPole-v1', render_mode="human")
        else:
            env.reset()

        for i_episode in range(200):
            observation = env.reset()
            for t in range(100):
                if self.log:
                    env.render()
                #print(observation)
                action = env.action_space.sample()
                
                observation, reward, done, truncated, info = env.step(action) # take a random action

                if done:
                    print("Episode finished later {} time steps".format(t+1))
                    break

        env.close()
    
    def build_training_set(self):
        self.env = gym.make("CartPole-v1")
        training_set = []
        score_set = []
        
        for game in range(self.nr_of_games_for_training):
            cumulative_game_score = 0
            game_memory = []
            obs_prev = (self.env.reset())[0]
            
            for step in range(self.game_steps):
                action = self.env.action_space.sample()
                obs_next, reward, done, _, info = self.env.step(action)
                game_memory.append([action, obs_prev])
                cumulative_game_score += reward
                obs_prev = obs_next
                
                if done:
                    print("Episode finished later {} time steps".format(step+1))
                    break
            
            if cumulative_game_score > self.score_requirement:
                score_set.append(cumulative_game_score)
                for play in game_memory:
                    if play[0] == 0:
                        one_hot_action = [1,0]
                    elif play[0] == 1:
                        one_hot_action = [0,1]
                    
                    training_set.append([one_hot_action, play[1]])
                
        with open("CartPoleTrainingSet.pickle", "wb") as f:
            pickle.dump(training_set, f)
            
        #pprint(training_set)
        if score_set:
            pass
            #print('Average Score: ' + mean(score_set))
        return training_set
            
    def play_game(self):
        f = open('CartPoleTrainingSet.pickle', 'rb')
        df = pickle.load(f)
        
        # alterar este workflow
        mlp = MLP(20, 32, 2)
        mlp.build()
        training_set = mlp.prepare_data(df)
        mlp.fit(training_set)
        
        if self.log:
            self.env = gym.make("CartPole-v1", render_mode="human")
        else:
            self.env = gym.make("CartPole-v1")
        
        for game in range(self.nr_of_games_for_training):
            score = 0
            
            obs_prev = (self.env.reset())[0]
            done = False
            
            while not done:
                if self.log:
                    self.env.render()

                action = mlp.predict(obs_prev)                
                #print("ACTION: ", action)
                obs_next, reward, done, _, _ = self.env.step(action)
                score += reward
                obs_prev = obs_next
                print('Current score: ' + str(score))
            
            self.scores.append(score)
            print('Current score: ' + str(score) + ';Average scores', sum(self.scores)/len(self.scores))
        print("Max score: ",max(self.scores))



class MLP():
    def __init__(self, epochs=1, batch_size=32, output_neurons=2):
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_neurons = output_neurons
        self.model = None
        
    def prepare_data(self, training_dataset):
        X = np.array([i[1] for i in training_dataset], dtype="float32").reshape(-1, len(training_dataset[0][1]))
        y = np.array([i[0] for i in training_dataset])
        pprint(training_dataset)
        return X, y
    
    def build(self):
        """
        Build the model
        """
        self.model = tf.keras.Sequential([
            layers.Flatten(input_shape=(4,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(units=self.output_neurons, activation="softmax")
        ], name="PlayerAgent_Coach")
        
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy']
        )
            
    def fit(self, training_set):
        
        x_train, y_train = training_set
                
        print("Labels\n", y_train)
        history = self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            validation_split=0.1
        )

        #print('\nHistory values per epoch: ', history.history)
        # Faz falta adicionar o validation dataset    
    
    def predict(self, obs):
        #print("Value to be predicted: ", obs)
        obs = np.array(obs).reshape(1, -1)

        prediction = self.model.predict(obs)
        #print("Prediction: ", prediction)
        predicted_value = np.argmax(prediction)
        #print(predicted_value)
        
        return predicted_value