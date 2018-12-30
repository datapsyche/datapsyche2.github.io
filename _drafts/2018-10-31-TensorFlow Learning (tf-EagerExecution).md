y So this is a walk through of all the concepts that i have tried to learn in past 2 days about tensorflow. There are 4 highlevel API's that were introduced in the latest version of TensorFlow (1.9 as on 10-31-2018). lets discuss about second of those 4 high level API's, first being tf.keras.



### Part-2 - Eager Execution

so this is relatively something that tensorflow borrowed from pytorch, in eager execution mode tensorflow just gives away the graphs and sessions paradigm. Here operations are evaluated immediately without building graphs. This makes tensorflow easier to understand as well as debug. The tensorflow code now resembles much more closer to native python code when in Eager execution mode. All the updated tensorflow libraries have the option to enable eager execution. below command quicly enables eager execution.

```python
import tensorflow as tf
tf.enable_eager_execution()

tf.executing_eagerly() ## returns True in eage_execution mode
```

so how does this eager execution works. ? simple answer is it works just like python

```python
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))
## output -> "hello, [[4.]]"
```

**With Numpy**

Eager execution works closely with numpy. numpy operations accepts `tf.Tensor` arguments. `tf.Tensor.numpy` method returns objects value in numpy.

```python
a = tf.constant([[1, 2],[3, 4]])
print(a)
#output => tf.Tensor([[1 2] [3 4]], shape=(2, 2), dtype=int32)

b = tf.add(a, 1)
print(b)
#output => tf.Tensor([[2 3] [4 5]], shape=(2, 2), dtype=int32)

print(a * b)
#output => tf.Tensor([[ 2  6] [12 20]], shape=(2, 2), dtype=int32)

import numpy as np

c = np.multiply(a, b)
print(c)
#output => [[ 2  6] [12 20]]

print(a.numpy())
#output => [[1 2] [3 4]]
```

`tf.contrib.eager` module contains symbols available to both eager  and graph execution environments and is useful for writing code to work with graph. Hence we could write If and for loops just like a python variable for a tensorflow variable.

**Building a Model**

When using tensorflow with eager execution we can write our own layers or use a layer provided in the `tf.keras.layers` package. `tf.keras.layers.Layers` can be used as our Base class and inherit from this base class to implement our own custom layer.

```python
class MySimpleLayer(tf.keras.layer.Layer):
    def __init__(self, output_units):
        super(MySimpleLayer,self).__init__()
        self.output_units = output_units
        
    def build(self,input_shape):
        self.kernel = self.add_variable("kernel",[input_shape[-1],self.output_units])
    
    def call(self,input):
        return tf.matmul(input, self.kernel)
```

So we have a custom layer ready, now we could prepare our model, lets go with the functional way of configuring a model.

```python
class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel,self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=10)
        self.dense2 = tf.keras.layers.Dense(units=10)
        
    def call(self,input):
        result = self.dense1(input)
        result = self.dense2(result)
        result = self.dense2(result)
        return result

model = MNISTModel()
```

We have a model ready now. lets get to train our model, for that we need to know how to compute the gradient.



**Computing Gradient**

Automatic differentiation is useful for implementing machine learning algorithms such as backpropagation for training neural networks. During eager execution, use `tf.GradientTape` to trace operations for computing gradients later. `tf.GradientTape` is an opt in feature to provide maximal performance when not tracing. To compute gradient we play the tape backwards and then discard. A particular `tf.GradientTape` can only compute one gradient subsequent calls throws error.

```python
w=tf.Variabel([[1.0]])
with tf.GradientTape() as tape:
    loss = w*w
grad = tape.gradient(loss,w)
print(grad)
#Output-> tf.Tensor([[2.]]
```

lets look into this concept and how it is applied in a simple deep learning scenario.

```python
NUM_EXAMPLES=1000
training_inputs=tf.ranndom_normal([NUM_EXAMPLES])
noise=tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs*3+2+noise

def prediction(input,weight,bias):
    return input*weight+bias

def loss(weights,bias):
    error = prediction(training_inputs,weights,bias)-training_outputs
    return tf.reduce_mean(tf.square(error))

def grad(weigths, bias):
    with tf.GradientTape() as tape:
        loss_value = loss(weights, biases)
    return tape.gradient(loss_value,[weights, biases])

train_steps = 200
learning_rate =0.01

W=tf.Variable(5.)
W=tf.Variable(10.)

print("Initial Loss : {:.3f}".format(loss(W,B)))

for i in range(train_steps):
    dW, dB = grad(W,B)
    W.assign_sub(dW*learning_rate)
    W.assign_sub(dB*learning_rate)
    if i%20==0:
        print("Loss at Step {:03d}:{:.3f}".format(i, loss(W,B)))
print("Final loss: {:.3f}".format(loss(W,B)))
print("W={},B={}".format(W.numpy(),B.numpy()))
```

This should be able to help you understand how to write a simple regression model in tensorflow. now lets look into writing a simple classification problem using vanila tensorflow.

**Classification** - MNIST Digit dataset

```python
import dataset
dataset_train = dataset.train('./datasets').shuffle(60000).repeat(4).batch(32)

def loss(model, x, y):
    prediction = model(x)
    return tf.losses.sparse_softmax_crossentropy(labels=y, logits=prediction)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model,inputs,targets)
    return tape.gradient(loss_Value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

x,y = iter(dataset_train).next()
print("Initial loss: {:.3f}".format(loss(model, x, y)))

for (i ,(x,y)) in enumerate(dataset_train):
    grads=grad(model, x, y)
    optimizer.apply_gradients(zip(grads,model.variables), global_step = tf.train.get_or_create_global_step())
    if i%200==0:
        print("Loss at step {:.4d}:{:.3f}".format(i, loss(model, x, y)))
    print("Final loss : {:.3f}".format(loss(model, x, y)))    
```

we could also move the computation to GPU for faster training.

```python
with tf.device("/gpu:0"):
    for (i ,(x,y)) in enumerate(dataset_train):
        optimizer.minimize(lambda:loss(model, x, y), global_step = tf.train.get_or_create_global_step())
```



**Variables and Optimizers**

`tf.Variable` object stores mutable `tf.Tensor` values accessed during training to make automatic differentiation easier. The parameters of a model can be encapsulated in classes as `tf.Variables` with `tf.GradientTape`. lets try this out.

```python
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.W = tf.Variable(5., name='weight')
        self.B = tf.Variable(10., name='bias')
    def call (self, inputs):
        return inputs*self.W +self.B
    
    NUM_EXAMPLES = 2000
    training_inputs = tf.random_normal([NUM_EXAMPLES])
    noise = tf.random_normal([NUM_EXAMPLES])
    training_outputs = training_inputs*3 + 2 + noise
    
    def loss(model, inputs, targets):
        error = model(inputs) - targets
        return tf.reduce_mean(tf.square(error))
    
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets)
        return tape.gradient(loss_value, [model.W, model.B])
    
    model = Model()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    print("Initial loss : {:.3f}".format(loss(model, training_inputs, training_outputs)))
    
    for i in range(300):
        grads = grad(model, training_inputs, training_outputs)
        optimizer.apply_gradients(zip(grads, [Model.W, model.B]),global_step=tf.train.get_or_create_global_step())
        if i % 20==0:
            print("Loss at step {:03d}: {:.3f}".format(i,loss(model, training_inputs, training_outputs)))
    print("Final Loss : {:.3f}".format(loss(model, training_inputs, training_outputs)))
    print("W = {}, B = {}".format(model.W.numpy(),model.B.numpy()))        
```

**Object Based Saving**

`tf.train.Checkpoint` can save and restore `tf.Variable` to and fro from checkpoints.

```python
x = tf.Variable(10.)
checkpoint = tf.train.Checkpoint(x=x)
x.assign(2.)
save_path = checkpoint.save('./ckpt/')
x.assign(11.)
checkpoint.restore(save_path)
print(x)
#output -> 2.0
```

**To save and load Model through Checkpoints**

to record the state of a model an optimizer and a global step we need to pass them to a tf.train.Checkpoint stores the internal state of objects, without requiring hidden variables. lets try this out.

```python
model = MyModel()
optimizer = tf.train.AdamOptimizer(learning_rate=.001)
checkpoint_dir = '/path_to_model_dir'
checkpoint_prefix = os.path.join(checkpoint_dir,"ckpt")
root = tf.train.Checkpoint(optimizer, model = model, optimizer_step = tf.train.get_or_create_global_step())
root.save(file_prefix = checkpoint_prefix) #or 
root.restore(tf.train.latest_checkpoint(checkpoint_dir))
```

**Object Oriented Metrics**

we could store metrics as a variable as shown below.

```python
m = tfe.metrics.Mean("loss")
m(0)
m(5)
m.result()  # output -> 2.5
m([8,9])
m.result()  # output -> 5.5
```

**Summaries and TensorBoard**

TensorBoard is a visualisation tool for understanding, debugging and optimising the model training process. it uses summary events to display it to the user.

`tf.contrib.summary` is compatible with both wager and graph execution environments. Summary operations such as `tf.contrib.summary.scalar` are inserted during model construction . lets see how to do this.

```python
global_step = tf.train.get_or_create_global_step()
writer = tf.contrib.summary.create_file_writer(logdir)
writer.set_as_default()
for _ in range(iterations):
    global_step.assign_add(1)
    with tf.contrib.summary.record_summaries_every_n_global_steps(100):
        tf.contrib.summary.scalar('loss',loss)
        ....
        
```



### Automatic Differentiation : Advanced Concepts

**Dynamic Model** - `tf.GradientTape` can also be used with dynamic model.

```python
def line_search_step(fn, init_x, rate =1.0):
    tape.watch(init_x)
    value = fn(init_x)
grad = tape.gradient(value, init_x)
grad_norm = tf.reduce_sum(grad*grad)
init_value = value
while value > init_value -rate*grad_norm
	x = init_x - rate*grad
    value = fb(x)
    rate /=2.0
 return x, value

```

Like `tf.GradientTape` there are other major functions to compute gradients some of them are discussed below. These functions are usefull for writing math code with only tensor and gradient functions and without `tf.Variables`

`tfe.gradients_function` -  Returns a function that computes the derivatives of its input function parameter with respect to its arguments.

`tfe.value_and_gradients_function`  - simialar to `tfe.gradients_function`  it returns the value from the input function in addition to the list of derivatives of the input function with respect to its arguments.

lets work on some examples 

```python
def square(x):
    return tf.multiply(x,x)
grad = tfe.gradients_function(square)
square(3.) # output -> 9.0
grad(3.)   # output -> [6.0]

gradgrad = tf.gradients_function(lambda x:grad(x)[0])
grad(3.)  #output -> [2.0]

gradgradgrad = tfe.gradients_function(lambda x: gradgrad(x)[0])
gradgradgrad(3.) #output -> [None]

def abs(x):
    return x if x > 0. else -x

grad = tfe.gradients_function(abs)
grad(3.)   #output -> [1.0]
grad(-3.)  #output -> [-1.0]

```

**Custom Gradient**

lets consider below example . 

``` python
def log1pexp(x):
    return tf.log(1+tf.exp(x))
grad+log1pexp = tfe.gradients_function(log1pexp)

grad_log1pexp(0)  #output -> [0.5]
grad_log1pexp(100) #output -> [nan]
# x=100 fails because of numerical instability.
```

Now let us create a custom gradient for above function

```python
@tf.custom_gradient
def log1pexp(x):
    e = tf.exp(x)
    def grad(dy):
        return dy * (1-1 / (1+e))
    return tf.log(1+e), grad

grad_log1pexp = tfe.gradients_function(log1pexp)

#As before, the gradient computation works fine at x=0
grad_log1pexp(0.) #output -> [0.5]

# and the gradient computation also works at x=100
grad_log1pexp(100.) #output-> [1.0]
```

### Performance

In eager execution computation is automatically offloaded to GPU. however this could be controlled using `tf.device('/gpu:0')` or `tf.device('cpu:0')` command as per necessity. lets try this 

```python
import time
def measure (x, steps):
    tf.matmul(x,x)
    start = time.time()
    for i in range(steps):
        x = tf.matmul(x,x)
    _ = x.numpy()
    end = time.time()
    return end - start
shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matric by itself {} times :".format(shape, steps))

# run on CPU:
with tf.device("/cpu:0"):
    print("CPU:{} secs".format(measure(tf.random_normal(shape),steps)))
    
# run on GPU, if available:
if tfe.num_gpus()>0:
    with tf.device("/gpu:0"):
        print("GPU:{} secs".format(measure(tf.random_normal(shape),steps)))
        else:
            print("GPU: not found")      
```

Lets also try to compute some operation in GPU while part in CPU.

```python
x = tf.random_normal([10,10])
x_gpu0 = x.gpu()
x_cpu = x.cpu()
_ = tf.matmul(x_cpu,x_cpu)
_ = tf.matmul(x_gpu0,x_gpu0)

if tfe.num_gpus() > 1:
    x_gpu1=x.gpu(1)
    _ = tf.matmul(x_gpu1,x_gpu1)
```



**Working with Graphs**

eager execution makes development and debugging more interactive. But TensorFlow graph execution does have some advantages like distributed training, performance optimisations and production deployment. But writing a graph code is different from python code and it is quite difficult to decode for  a student programmer.  an eager execution code  will also run in tensorflow graph execution, the only difference would be that that we wont have to use `tf.enable_eager_execution()` in the beginning of our session. As per the tensorflow guide the best way to write a tensorflow program is to write the code parallely in eager execution mode and graph mode. test and debug in eager execution mode while run and deploy in graph mode.

**Using eager execution in Graph mode**

below example selectively enable eager execution in a tensorflow graph environment using  `tfe.py_func` interesting thing here is we have not used `tf.enable_eager_execution()` at all.

```python
def my_py_func(x):
    x = tf.matmul(x,x)
    print(x)
    return x

with tf.Session() as sess:
    x = tf.placeholder (dtype = tf.float32)
    # call eager function in graph!
    pf = tfe.py_func(my_py_func,[x], tf.float32)
    sess.run(pf, feed_dict = {x :[[2.0]]})
#output ->  [[4.0]]
```




















