So this is my attempt to learn the very basics of tensorflow. I'm currently following a book **Fundatmentals of Deep Learning** by *Nikhil Buduma*. It is available in the public domain and provides an excellent intuition of neural networks in TensorFlow. I've been trying to master TensorFlow for previous couple of weeks and i decided to start with a book rather than learning online. 

Thankfully I'm an expert in installing tensorflow in Anaconda i've done it multiple times.  lets start with the next basic step. 



**Import tensorflow**

`import tensorflow as tf`    # This import tensorflow with tf as an alias.



**Hello World**

let's print 'hello World' in tensorflow

```python
import tensorflow as tf
helloWorld= tf.constant('Hello World')
sess = tf.session()
session.run(helloWorld)
----> 'Hello World'
```

bit length when compared to python. Interestingly as per latest tensor flow version, we have something called eager execution where in we can do away with tf.Session()

**Constants**

```python
import tensorflow as tf
tf.enable_eager_execution()
const=tf.constant('Hello World')
print(const)
---->tf.Tensor(b'Hello World', shape=(), dtype=string)
```

I'm still learning the nuances of tensorflow enable_eager execution. I shall write more on this  when i have a clear picture about it. lets go back to the graph and sessions paradigm once again.

```python
a = tf.constant(2)
b = tf.constant(3)
mul = tf.multiply(a,b)
session.run(mul)
----> 6
```

**Variables**

* Variables must be explicitly initialized before a graph is used for the first time
* Gradient Method could be used to modify the variables after each iteration
* We can save values stored in variables to disk and restore them for later use.

```python
weights = tf.Variable(tf.random_normal([300,200],stddev=0.5), name='weights', trainable=False)
```

* tf.random_normal is an operation that produce a tensor initialised using a normal distribution with standard deviation of 0.5, we could use other initializers as well they are

  ``` python
  tf.zeros(shape, dtype=tf.float32)
  tf.ones(shape, dtype=tf.float32)
  tf.random_normal(shape, dtype=tf.float32,mean=0, stddev=1,seed=0)
  tf.truncated_normal(shape, dtype=tf.float32,mean=0, stddev=1,seed=0)
  tf.random_uniform(shape, dtype=tf.float32,mean=0, stddev=1,seed=0)
  ```

* tensor shape is 300x200, implying weights connect a layer with 300 neurons to a layer with 200 neurons

* we have also passed a name to the variable 'weights'

* trainable = True /False lets the tensorflow know that if the weights are meant to be trainable.

