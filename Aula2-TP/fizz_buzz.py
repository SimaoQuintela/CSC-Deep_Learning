import tensorflow as tf
import numpy as np

print(tf.executing_eagerly())

def fizzbuzz(limit):
    print('is limit a tensor? %s' %tf.is_tensor(limit))
    
    if(not tf.is_tensor(limit)):
        limit = tf.convert_to_tensor(limit)
        print('Is it a tensor now? %s' %tf.is_tensor(limit))
        
        
    three = tf.constant(3)
    five = tf.constant(5)
    
    for i in tf.range(1, limit+1):
        if(tf.math.mod(i, three) == 0 and tf.math.mod(i, five) == 0):
            print("Fizz Buzz")
        elif(tf.math.mod(i, three) == 0):
            print("Fizz")
        elif(tf.math.mod(i, five) == 0):
            print("Buzz")
        else:
            print(i.numpy())
        
        
    
fizzbuzz(tf.constant(15))