import tensorflow as tf
import matplotlib.pyplot as plt

#tensorflow version being used
print(tf.__version__)
#is tf executing eagerly?
print(tf.executing_eagerly())

# load mnist training and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# data shape and cardinality
print('Train data shape', x_train.shape, y_train.shape)
print('Test data shape', x_test.shape, y_test.shape)

print('Number of training samples', x_train.shape[0])
print('Number of testing samples',  x_test.shape[0])


"""
#plotting some numbers!
for i in range(2):
    #Add a subplot as 5 x 5
    plt.subplot(5,5,i+1)
    #get rid of labels
    plt.xticks([])
    #get rid of labels
    plt.yticks([])
    
    plt.imshow(x_test[i], cmap="gray")
    plt.show()
"""


# reshape the input to have a list of 784 (28*28) and normalize it (/255)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_train = x_train.astype('float32')/255

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_test = x_test.astype('float32')/255

# build a three-layer sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compiling the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# training it
model.fit(x_train, y_train, epochs=5)

# evaluating it
_, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)

# finnaly, generating predictions (the output of the last layer)
print('\nGenerating predictions for the first fifteen layers')
predictions = model.predict(x_test[:15])
print('Predictions shape:', predictions.shape)

for i, prediction in enumerate(predictions):
    predicted_value = tf.argmax(prediction)
    label = y_test[i]
    print('Predicted a %d. Real value is %d.' % (predicted_value, label))
