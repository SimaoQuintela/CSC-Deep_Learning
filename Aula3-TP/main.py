import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from perceptron_layer import MultiLayerPerceptron

logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.random.set_seed(3192873128)
# for an easy reset backend session state
tf.keras.backend.clear_session()



def import_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # some data exploration
    print('------- Log import_data -------')
    print('Train data shape', x_train.shape)
    print('Test data shape', x_test.shape)
    print('Number of training samples', x_train.shape[0])
    print('Number of testing samples', x_test.shape[0])
    
    '''
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_test[i], cmap='gray')
    plt.show()
    '''
    
    # reshape the input to have a list of self.batch_size by 28*28 = 784, and normalize (/255)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]).astype('float32')/255
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]).astype('float32')/255
    
    # reserve 5000 samples for validation
    x_validation = x_train[-5000:]
    y_validation = y_train[-5000:]
    
    # do not use the same 5000 samples for training!
    x_train = x_train[:-5000]
    y_train = y_train[:-5000]

    # create dataset iterator for training
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # shuffle in intervals of 1024 and slice in batchs of batch size
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    # create the validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))
    validation_dataset = validation_dataset.batch(batch_size)
    return train_dataset, validation_dataset, x_test, y_test
    

'''
Preparing the model, the optimizers, the loss function and some metrics.
'''
def prepare_model():
    mlp = MultiLayerPerceptron(output_neurons=output_neurons)
    # instantiate an optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # instantiate a loss object (from_logits=False as we are applying a softmax activation over the last layer)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    return mlp, optimizer, loss_object, train_metric, val_metric

'''
Define a low level fit and predict making use of the tape gradient
'''
def low_level_fit_and_predict():
    # manually, let's iterate over the epochs and fit ourselves
    for epoch in range(epochs):
        print('Epoch %d %d' % (epoch+1, epoch))
        
        # to store loss values
        loss_history = []
        
        # iterate over all batchs
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            # use a gradient tape to save computations to calculate gradient later
            with tf.GradientTape() as tape:
                # running the forward pass of all layers
                # operations being recorded into the tape
                probs = mlp(x_batch)
                # computing the loss for this batch
                # how far are we from the correct labels?
                loss_value = loss_object(y_batch, probs)
            
            #store loss value
            loss_history.append(loss_value.numpy().mean())
            # use the tape to automatically retrieve the gradients of the trainable variables
            # with respect to the loss
            gradients = tape.gradient(loss_value, mlp.trainable_weights)
            # running one step of gradient descent by updating (going backwards now)
            # the value of the trainable variables to minimize the loss
            optimizer.apply_gradients(zip(gradients, mlp.trainable_weights))
            # Update training metric
            train_metric(y_batch, probs)
            
            if step%200 == 0:
                print('Step %s; Loss Value = %s; Mean Loss = %s' %(step, str(loss_value.numpy()), np.mean(loss_history)))
                
        # display metrics at the end of each epoch
        train_accuracy = train_metric.result()
        print('Training accuracy for epoch %d: %s' %(epoch +1, float(train_accuracy)))
        
        # reset training metrics (at the end of each epoch)
        train_metric.reset_states()
        
        # run a validation loop at the end of each epoch
        for x_batch_val, y_batch_val in validation_dataset:
            val_probs = mlp(x_batch_val)
            # update val metrics
            val_metric(y_batch_val, val_probs)
        val_acc = val_metric.result()
        val_metric.reset_states()
        print('Validation accuracy for epoch %d: %s' % (epoch +1, float(val_acc)))
    
    # now predict
    print('\nGenerating predictions for ten samples....')
    predictions = mlp(x_test[:10])
    print('Predictions shape: ', predictions.shape)
    
    for i, prediction in enumerate(predictions):
        predicted_value = tf.argmax(prediction)
        label = y_test[i]
        print('Predicted a %d. Real value is %d.' %(predicted_value, label))            


'''
Define a high level fit and predict making use of the tape gradient
aka what we did last class
'''
def high_level_fit_and_predict():
    # shortcut to compile and fit a model
    mlp.compile(optimizer='adam', loss=loss_object, metrics=[train_metric])
    # since the train_dataset already takes care of batching, we don't pass a batch_size argument
    # passing validation data for monitoring validation loss and metrics at the end of each epoch
    history = mlp.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)
    print('\nHistory values per epoch: ', history.history)
    
    # evaluating the model on the test data
    print('\nEvaluating model on test data...')
    scores = mlp.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print('Evaluation %s: %s' %(mlp.metrics_names, str(scores)))
    
    # finally, generating predictions (the output of the last layer)
    predictions = mlp.predict(x_test[:10])
    print('Predictions shape: ', predictions.shape)

    for i, prediction in enumerate(predictions):
        # tf.argmax returns the INDEX with largest value across axes of a tensor
        predicted_value = tf.argmax(prediction)
        label = y_test[i]
        print('Predicted a %d. Real value is %d.' %(predicted_value, label))


# hyperparameters
epochs = 5
batch_size = 32
learning_rate = 1e-3
output_neurons = 10

# load data
train_dataset, validation_dataset, x_test, y_test = import_data()

# init our model
mlp, optimizer, loss_object, train_metric, val_metric = prepare_model()

# use low-level or high-level fit and predict
low_level_fit_and_predict()
#high_level_fit_and_predict()