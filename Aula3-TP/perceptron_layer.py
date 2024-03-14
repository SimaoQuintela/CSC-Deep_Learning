import tensorflow as tf
from tensorflow.python.keras.layers import Layer 
from tensorflow.python.keras import Model

# This class extends the class Layer
class PerceptronLayer(Layer):
    def __init__(self, neurons=16, **kwargs):
        super(PerceptronLayer, self).__init__(**kwargs)
        self.neurons = neurons
        
    
    '''
    We use the build function to deferr weight creation until the shape of the inputs is known.
    '''
    def build(self, input_shape):
        # set the weights
        self.w = self.add_weight(shape=(input_shape[1], self.neurons), initializer='random_normal', trainable=True)
        # set the bias
        self.b = self.add_weight(shape=(self.neurons,), initializer='random_normal', trainable=True)


    '''
    Implements the function call operator (when an instance is used as a function).
    It will automaitcally run build the first time it is called, i.e, layer's weights are created dinamically.
    '''
    def call(self, inputs):
        # return the perceptron result using the linear equation, Z = x * w + b , where x is the input
        return tf.matmul(inputs, self.w) +  self.b
        
    
    
    '''
    Enable serialization on our perceptron layer
    '''
    def get_config(self):
        config = super(PerceptronLayer, self).get_config()
        config.update({'neurons': self.neurons})
        return config
    
    
# This class extends the class Model
class MultiLayerPerceptron(Model):
    
    '''
    The Layers of our MLP (whith a fixed number of neurons)
    '''
    def __init__(self, output_neurons=10, name='multilayerPerceptron', **kwargs):
        super(MultiLayerPerceptron, self).__init__(name=name, **kwargs)
        self.perceptron_layer1 = PerceptronLayer(16)
        self.perceptron_layer2 = PerceptronLayer(32)
        self.perceptron_layer3 = PerceptronLayer(output_neurons)
        
    '''
    Layer are recursively composable, i.e,
    if you assign a Layer instance as attribute of another Layer, the outer Layer will start tracking the weights of the inner Layer.
    Remember that the build of each Layer is called automatically (thus creating the weights)
    '''
    def feed_model(self, input_data):
        x = self.perceptron_layer1(input_data)
        # activation function applied to the output of the perceptron layer
        x = tf.nn.relu(x)
        
        # the output, now normalized, is fed as input to the second perceptron layer
        x = self.perceptron_layer2(x)
        # again, activation function applied to the output of the second perceptron layer
        x = tf.nn.relu(x)
        
        # again, fed x as input to the third and last layer.
        logits = self.perceptron_layer3(x)
        
        # the output of the last layer going over a softmax activation
        # so, we will not be outputtinh logits but "probabilities"
        return self.softmax(logits)

        
    '''
    Compute softmax values for the logits
    '''
    def softmax(self, logits):
        sum_logits = sum([tf.math.exp(logits)])
     
        return ( tf.math.exp(logits)/(tf.math.reduce_sum(tf.math.exp(logits), axis=1, keepdims=True)) )
            
        
    def print_trainable_weights(self):
        print("Weights: ", len(self.weights))
        print('Trainable weights: ', len(self.trainable_weights))
        print('Non-trainable weights: ', len(self.non_trainable_weights))
        
        
    def call(self, input_data):
        # here, we want to feed the model and receive its output
        probs = self.feed_model(input_data)
        return probs