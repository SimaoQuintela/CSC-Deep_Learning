import gym
env = gym.make('CartPole-v1', render_mode="human")
env.reset()

for i_episode in range(20):
   observation = env.reset()
   for t in range(100):
      env.render()
      #print(observation)
      action = env.action_space.sample()
      observation, reward, done, _, info = env.step(action) # take a random action

      if done:
         print("Episode finished later {} time steps".format(t+1))
         break

env.close()