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

### Exercise - 2)  Basic Operations with Constant

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

### Exercise - 2)  Basic Operations with Variables

lets do the above exercise with variables now.

```python
a = tf.variable(tf.int16)
b = tf.variable(tf.int16)

add = tf.add(a,b)
mul = tf.multiply(a*b)

with tf.Session() as sess:
    print("Addition of Variables - {}".format(sess.run(add,feed_dict = {a:10,b:5})))
    print("Multiplication of Variables - {}".format(sess.run(mul,feed_dict = {a:10,b:5})))

##-> Output
#Addition of Variables : 15
#Multiplication of Variables : 50
```

### Exercise - 3) Matrix Multiplication

This is a straight forward example of matrix multiplication. please make sure the shape of the matrix is correct else chances of running into some shape error.

```python
mat1 = tf.constant([[3,3]])
mat2 = tf.constant([[2],[2]])
result = tf.matmul(mat1,mat2)
with tf.Session() as sess:
    print("Matrix Multiplication of ([3,3]) and ([2],[2]) = {}".format(sess.run(result))

#-> Output
# Matrix Multiplication of ([3,3]) and ([2],[2]) = [[12]]
```

### Exercise - 4) Eager Execution

This is a relatively new paradigm in tensorflow programming. in eager execution mode we can do away with graph or interestingly we don't have to start a session to do tensorflow computations. This idea is actually  borrowed from the main competitor of tensorflow - pytorch. lets code it.

```python
import tensorflow as tf
print("Setting Eager Mode")
tf.enable_eager_execution()
print(tf.executing_eagerly())

#->Output
# Setting Eager Mode
# True
```

My Observations

* Eager Execution should be enabled at the start of the program, If you are in the middle of a Jupyter notebook i request you to open a new notebook for this exercise, else it will throw an error message. I hope the next version of tensorflow (tensorflow 2.0) would fix this issue and eager execution will be the default setting.

Lets get back to code.

```python
a = tf.constant(5)
b = tf.constant(10)
print("Tensor Constant a : {}".format(a))
print("Tensor Constant b : {}".format(b))
print("Tensor Constant a+b : {}".format(a+b))

#-> Output
# Tensor Constant a : 5
# Tensor Constant b : 10
# Additon of Tensor constant a+b : 15
```

Observations

* We now have sessions.
* The Code now looks more pythonic 

```python
import numpy as np
a = tf.constant([[2,3],[2,4]],dtype=tf.float32)
b = np.array([[1,2],[3,2]],dtype=np.float32)
print("Addition : {}".format(a+b))
print("Subtraction : {}".format(a-b))
print("Multiplication : {}".format(a*b))

#->Output
#Tensor Additon - [[3. 5.]
# [5. 6.]]
#Tensor Subtraction- [[ 1.  1.]
# [-1.  2.]]
#Tensor Multiplication- [[2. 6.]
# [6. 8.]]
```

