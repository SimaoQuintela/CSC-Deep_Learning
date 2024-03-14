import tensorflow as tf
import numpy as np

#tf.random.set_seed(1234)

def um_a():
    a = tf.random.uniform([])
    b = tf.random.uniform([])
    
    if(a<b):
        res = tf.add(a, b)
    else:
        res = tf.subtract(a,b)
    
    print("a", a)
    print("b", b)
    print("res", res)
    
    return res
    
def um_a():
    a = tf.random.uniform([], -1, 1)
    b = tf.random.uniform([], -1, 1)
    
    if(a<b):
        res = tf.add(a, b)
    elif(a>b):
        res = tf.subtract(a,b)
    else:
        res = 0
    
    print("a", a)
    print("b", b)
    print("res", res)
    
    return res    

def um_c():
    a = tf.Variable([[1, 2, 0], [3, 0, 2]])
    b = tf.zeros_like(a)
    
    print("a", a)
    print("b", b)
    
    equals = tf.math.equal(a, b)
    print("equals", equals)
    
    return

def um_d():
    a = tf.random.uniform([20], minval=1, maxval=10, dtype=tf.dtypes.int64)
    out = tf.gather(a, tf.where(a>7))
    print("a", a)
    print("out",out)

#print(um_c())
 

       