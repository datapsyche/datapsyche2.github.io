 So this is a walk through of all the concepts that i have tried to learn in past 2 days about tensorflow. There are 4 highlevel API's that were introduced in the latest version of TensorFlow (1.9 as on 10-31-2018). lets discuss about each of those 4 high level API's.

### Keras 

main features of Keras are *User Friendly, Modular and Composable, Easy to extend* . Lets dive into code. 


``` python
import tensorflow as tf
from tensorflow import keras
```


so keras is now being shipped with tensorflow, now there is no need to separately install keras this helps as original keras has multiple backends like theano, tensorflow etc. well now we have keras with just tensorflow and hence lightweight. `tf.keras.version` could be helpfull incase if you want to know which version of keras are we working with (helpfull when working with prebuild models).

#### Building a model

the methodolgy is exactly similar to how models were built using keras. 

```python
model =keras.Sequential()
model.add(keras.layers.Dense(64,activation='relu',kernel_regularizer=keras.regularizers.l1(0.01)))
model.add(keras.layers.Dense(64,activation='relu',bias_initializer=keras.initializers.constant(2.0)))
model.add(keras.layers.Dense(10,activation='softmax'))
```

`Dense` is just one of the layers available in keras.layers there are others like `conv2d` `maxpooling`,`lstm`, etc available and which could be called upon based on the requirements. Similarly for activation also we have multiple functions availble apart from `relu` and `softmax` like `sigmoid`,`tanh`  etc. `keras.layers` takes in multiple parameters depending upon the type of layer. Some of the important parameters which are common to most layers are `number of neurons`,`kernal_initializer`and`bias_initializer`,`kernal_regularizer`  and  `bias regularizer`. Depending upon the layer there are will be other parameters.

``` python
model.compile(optimizer=tf.train.AdamOptimizer(0.001),loss='categorical_crossentropy',metrics=['accuracy]'])
```

optimizer again has multiple methods like `AdamOptimizer`,`RMSPropOptimzer` or `GradientDescentOptimizer`. loss is the function to minimize during optimization some common methods are `categorical_crossentropy` ,`binary_crossentropy`, `mse`.

metrics is used to monitor how the training is proceeding, gives an idea if our model is improving `accuracy`,  `precision`, `mae` etc are some of the important metrics that are used generally. 

so we have our model structure ready, lets feed it with data for training.

``` python
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))


model.fit(data, labels, epochs=10, batch_size=32)
```

`model.fit` takes three important arguments they are epochs (one iteration over the entire input data) , batch_size (model slices the data into smaller batches and iterates over these batches during training), validation_data (want to easily monitor its performance on some validation data. Passing this argument—a tuple of inputs and labels—allows the model to display the loss and metrics in inference mode for the passed data, at the end of each epoch)

#### Input Data using tf.data.Dataset

we can pass tf.data.Dataset from Datasets API instead of passing data to the model.fit method.

``` python
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()
model.fit(dataset, epochs=10, steps_per_epoch=30)
```

Here the Dataset yeilds batches of data hence batch_size is not required, steps_per_epoch is the number of training steps the model runs before it moves to the next epoch.

#### Evaluate and Predict

To evaluate and predict  we make use of Model.evaluate and Model.predict methods. 

``` python
model.evaluate(x, y, batch_size=32) # using numpy 
model.evaluate(dataset, steps=30) #using datasets
```

``` python 
model.predict(x, batch_size=32) # using numpy
model.predict(dataset, steps=30) #dataset
```

#### Functional API for building models

lets try out the functional API method for building a model instead of `keras.Sequential()` method.

```python
inputs = keras.Input(shape=(32,)) 

x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
predictions = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001), loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)
```

Now lets go little more deeper, with Model Subclassing. lets build a fully customizable model by subclassing keras. Model with a custom forward pass

#### Custom Models

```python
class MyModel(keras.Model):

  def __init__(self, num_classes=10):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    self.dense_1 = keras.layers.Dense(32, activation='relu')
    self.dense_2 = keras.layers.Dense(num_classes, activation='sigmoid')

  def call(self, inputs):
    x = self.dense_1(inputs)
    return self.dense_2(x)

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes
    return tf.TensorShape(shape)


model = MyModel(num_classes=10)
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)
```



#### Custom Layer

Now let us try to create the same model with a custom layer,  a custom layer can be created by subclassing tf.keras.layers.Layers with below methods. `build` - to create the weights of the layer and `.add_weight` method helps us to add a weight to layer, `call` defines the forward pass, `compute_output_shape` specify how to compute the shape of the output of layer, `get_config`method helps in serializing the data.

```python	
class MyLayer(keras.layers.Layer):

  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(MyLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    shape = tf.TensorShape((input_shape[1], self.output_dim))
    self.kernel = self.add_weight(name='kernel',
                                  shape=shape,
                                  initializer='uniform',
                                  trainable=True)
    super(MyLayer, self).build(input_shape)

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.output_dim
    return tf.TensorShape(shape)

  def get_config(self):
    base_config = super(MyLayer, self).get_config()
    base_config['output_dim'] = self.output_dim
    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


model = keras.Sequential([MyLayer(10),
                          keras.layers.Activation('softmax')])
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, targets, batch_size=32, epochs=5)
```



#### Callbacks

A callback is an object passed to a model to customize and extend its behaviour during training some of the widely used callbacks are 

`tf.keras.callbacks.ModelCheckpoint` - to save checkpoint of the model at regular interval.

`tf.keras.callbacks.LearningRateScheduler` : to change the learning rate

`tf.keras.callbacks.EarlyStopping` : Interrupt training when validation performance has stopped.

`tf.keras.callbacks.TensorBoard` :  Monitor model's behavious using tensorboard.

Sample of callbacks 

``` Python
callbacks=[keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
           keras.callbacks.TensorBoard(log_dir='./logs')]
model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks, validation_data=(val_data,val_targets))
```



#### Save and Restore a Model

`tf.keras.Model.save_weights` is used to save a model

```python
model.save_weights('./my_model')
model.load_weights('my_model')
```

model weights are saved in Tensorflow checkpoint file format. this could be changed to HDF5 format.

```python
model.save_weights('my_model.h5', save_format='h5')
model.load_weights('my_model.h5')
```



#### Saving the configuration of a model

by serializes the model architecture without any weights.

```python
json_string = model.to_json() # model to json format
fresh_model = keras.models.model_from_json(json_string) #create a model with saved json string

yaml_string = model.to_yaml() #model to yaml string
fresh_model = keras.models.model_from_yaml(yaml_string) #create a model with saved yaml string
```



#### Saving and loading an entire model

Entire model can also be saved .

```python
# Create a trivial model
model = keras.Sequential([
  keras.layers.Dense(10, activation='softmax', input_shape=(32,)),
  keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, targets, batch_size=32, epochs=5)

model.save('my_model.h5')

model = keras.models.load_model('my_model.h5')
```



**Eager Execution** - tf.keras supports eager execution , eager execution is beneficial for  model sub classing and building custom layers 



**Estimators** - tf.estimators are used for training models  on distributed environments. A `tf.keras.Model` can be trained with `tf.estimator` API by converting the model to an tf.estimator.Estimator object with tf.keras.estimator.model_to_estimator

```python
model = keras.Sequential([layers.Dense(10,activation='softmax'),
                          layers.Dense(10,activation='softmax')])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

estimator = keras.estimator.model_to_estimator(model)
```



#### Multiple GPU's

`tf.keras` models can run in multiple GPU's using `tf.contrib.distribute.DistributionStrategy` we also need to convert the model to estimator object as explained above and then train the estimator.

```python
#create a simple model.
model = keras.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(10,)))
model.add(keras.layers.Dense(1, activation='sigmoid'))
optimizer = tf.train.GradientDescentOptimizer(0.2)
model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.summary()

# define an input dataset.
def input_fn():
  x = np.random.random((1024, 10))
  y = np.random.randint(2, size=(1024, 1))
  x = tf.cast(x, tf.float32)
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.repeat(10)
  dataset = dataset.batch(32)
  return dataset

# create a distribution strategy and then create a config file.
strategy = tf.contrib.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy)

# create an estimator instance
keras_estimator = keras.estimator.model_to_estimator(
  keras_model=model,
  config=config,
  model_dir='/tmp/model_dir')

# train the estimator
keras_estimator.train(input_fn=input_fn, steps=10)
```

This concludes my first part of tensor flow tutorial. I hope to revisit it soon once i get to learn more aspects on the Estimators or model saving techniques.


