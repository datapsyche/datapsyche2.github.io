So Hi, Again I'm trying to start fresh with Tensor Flow Basics - but this time with more code. I'll be almost copy pasting the code (best way to learn is to first copy, understand and then refactor, as per my way). 

### Exercise - 1) Hello World!

Simple straight forward Hello World in TensorFlow

```python
import tensorflow as tf
hello = tf.constant("Hello World!")
sess = tf.Session()
sess.run(hello)

##-> Output - b'Hello World!'
```

### Exercise - 2)  Basic Operations

So important things 

* we are opening a tf.Session() as sess and the indentation is important it means the sess is not closed.
* we use the python string formating to print sentences {} are just part of that. don't get tensed it is simple. request you to look into W3 schools to learn more about string formating. i Just did now and it was a great learning. :)

```python
a = tf.constant(5)
b = tf.constant(4)
with tf.Session() as sess:
    print("Constant a = {} \nConstant b = {}".format(sess.run(a),sess.run(b)))
    print("Addition a + b = {}".format(sess.run(a+b)))
    print("Multiplication a*b = {}".format(sess.run(a*b)))
    print("Subtraction a-b = {}".format(sess.run(a-b)))
    print("Division a/b = {}".format(sess.run(a/b)))
    
##-> Output
#Constant a = 5 
#Constant b = 4
#Addition a + b = 9
#Multiplication a*b = 20
#Subtraction a-b = 1
#Division a/b = 1.25
```







