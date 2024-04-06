import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import layers  

logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.random.set_seed(3192873128)
# for an easy reset backend session state
tf.keras.backend.clear_session()


def import_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    '''
    for i in range(15):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_test[i], cmap='gray')
    plt.show()
    '''
    
    # Data normalization
    x_train = x_train.astype('float32')/255    
    x_test = x_test.astype('float32')/255    
        
    # create dataset iterator and shuffle the dataset
    #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    #train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    
    return x_train, y_train, x_test, y_test
    
def fit_and_predict():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    
    history = model.fit(
        x_train, 
        y_train,
        epochs=epochs,
        validation_split=0.1
    )
    print('\nHistory values per epoch: ', history.history)

    print('Evaluating model on test data...')
    scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    print('Evaluation %s: %s' % (model.metrics_names, str(scores)))

# hyperparameters
epochs = 5
batch_size = 32
learning_rate = 1e-3
output_neurons = 10   

x_train, y_train, x_test, y_test = import_data()
fit_and_predict()
