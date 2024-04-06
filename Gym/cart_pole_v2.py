from PlayerAgent import PlayerAgent
import tensorflow as tf 

print(tf.executing_eagerly())

playerAgent = PlayerAgent(5000,10000,75, game='CartPole-v1', log=True)
    
#playerAgent.play_random_games()
#playerAgent.build_training_set()

playerAgent.play_game()