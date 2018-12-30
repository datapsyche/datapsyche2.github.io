So this is a walk through of all the concepts that i have tried to learn in past 3 days about tensorflow. There are 4 highlevel API's that were introduced in the latest version of TensorFlow (1.9 as on 10-31-2018). lets discuss about third of those 4 high level API's, we have already coverd tf.keras, EagerExecution previously. lets look into Importing Data.

### Importing Data

Here our aim is to build a robust data Pipeline and Tensorflow has came up with `tf.data` API to enable us build complex input pipelines from simple reuseable pieces. `tf.data` helps us to make it easy to deal with large amount of data, different format of data with complicated transformation.

`tf.data` API introduces two new abstractions to tensorflow : 

`tf.data.Dataset` - to represent a sequence of element, in which each element contain one or more Tensor object. two important methods that could be put to use are 				    `tf.data.Dataset.from_tensor_slices(` - to construct a dataset from one or more `tf.Tensor` objects.

`tf.data.Dataset.batch()`  helps us to apply transformation on `tf.data.Dataset` object.

Next method is `tf.data.Iterator` - it provides the main way to extract elements from a dataset.  The operation returned by `Iterator.get_next()` yields the next element of a Dataset when executed and acts as an interface between input pipeline code and model. 



**Fundamentals of Creating a Dataset and Iterator objects**

For a input data pipeline we need to have a source defined if the source is in tensor format then `tf.data.Dataset.from_tensors()` or `tf.data.Dataset.from_tensor_slices()` could be used. instead if the data is in disk in the recommended `TFRecord` format we can use `tf.data.TFRecordDataset` method. Once the `Dataset` object is ready  we can transform it into a new `Dataset` by chaining method calls on the `tf.data.Dataset` object.  Elemental transformation on this `Dataset` object could be done using `Dataset.map` and multi element transformation could be carried out using `Dataset.batch()`. As mentioned earlier we make use of `tf.data.Iterator` for consuming values from `Dataset` object. `tf.data.Iterator` has two important methods namely `Iterator.initializer` to reinitialize iterator state and `Iterator.get_next()` to get the next element or next batch of element from the dataset.

**Dataset Structure**

A dataset comprises of elements that each have the same structure.  An element contains one or more `tf.Tensor` object called components and each component has a `tf.DType` representing the type of the elements in the tensor and a `tf.TensorShape` representing the the static shape of each element. lets dive into code rather than explaining here.

```python
dataset1 = tf.data.Dataset.from_slices(tf.random_uniform([4,10]))
print(dataset1.output_types)   # output -> tf.float32
print(dataset1.output_shapes)  # output -> (10,)

dataset2 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4]), tf.random_uniform([4,100], maxval = 100, dtype =tf.int32)
print(dataset2.output_types)   # output -> (tf.float32, tf.float32)
print(dataset2.output_shapes)  # output -> ((),(100,))

dataset3 = tf.data.Dataset.zip((dataset1,dataset2))
print(dataset3.output_types)   # output -> (tf.float32,(tf.float32, tf.float32))
print(dataset3.output_shapes)  # output -> (10,((),(100,)))
```

some examples of `Dataset` transformation function

```python
dataset1 = dataset1.map(lambda x: ...)
dataset2 = dataset1.flat_map(lambda x,y: ...)
dataset3 = dataset1.filter(lambda x,(y,z): ...)
```

#### Creating an Iterator

there are multiple types of iterator namely, *one-shot, initializable, reinitializable and feedable*

**One-Shot Iterator**

```python
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
    value = sess.run(next_element)
    assert i ==value
```

**Initializable** -  iterator requires you to run an explicit `iterator.initializer` operation before using it. it enables us to *parameterize* the definition of the dataset, using one or more `tf.placeholder()`

```python
max_value = tf.placeholder(tf.int64,shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess.run(iterator.initializer, feed_dict = {max_value : 10})
for i in range(10):
    value = sess.run(next_element)
    assert i == value

sess.run(iterator.initializer, feed_dict = {max_value : 100})
for i in range(100):
    value = sess.run(next_element)
    assert i == value
```

**Reinitializable**  -  A reinitializable iterator can be initialized from multiple different `Dataset` objects. like training dataset and validation dataset. These pipelines will typically use different Dataset objects that have the same structure.

```python
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

for _ in range(20):
  # Initialize an iterator over the training dataset.
  sess.run(training_init_op)
  for _ in range(100):
    sess.run(next_element)

  # Initialize an iterator over the validation dataset.
  sess.run(validation_init_op)
  for _ in range(50):
    sess.run(next_element)
```

**Feedable** -  Feedable gives us much more flexibility to choose from multiple iterators for each call to `tf.Session.run`. It offers all the functionality of reinitializable iterator but doesnot requires us to initialize the iterator from start of a dataset, while switching between iterators.

```python
training_dataset = tf.data.Dataset.range(100).map(lambda x:x+tf.random_uniform([],-10,10,tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string(handle, training_dataset.output	_types, training_dataset.output_shapes)
next_element = iterator.get_next()

training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

while True:
    for _ in range(200):
        sess.run(next_element, feed_dict = {handle: training_handle})
    sess.run(validation_iterator.initializer)
    for _ in range(50):
        sess.run(next_element, feed_dict = {handle : validation_handle})
```

**Consuming Values from an Iterator**

`Iterator.get_next()` method returns one or more `tf.Tensor`objects that correspond to the symbolic next element of an iterator. Calling `Iterator.get_next()` does not immediately advance the iterator. Instead you must use the returned `tf.Tensor` objects in a TensorFlow expression, and pass the result of that expression to `tf.Session.run()` to get the next elements and advance the iterator. If the iterator reaches the end of the dataset, executing the `Iterator.get_next()` operation will raise a `tf.errors.OutOfRangeError` hence the best practice here is to use it inside a try except loop.

```python
sess.run(iterator.initializer)
while True:
  try:
    sess.run(result)
  except tf.errors.OutOfRangeError:
    break
```

**Iterators in Complex Datasets** - Consider a dataset as shown below (dataset3) each element of the dataset has a nested structure, the return value of `Iterator.get_next()` will be one or more `tf.Tensor`objects in the same nested structure :

```python
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()

sess.run(iterator.initializer)
next1, (next2, next3) = iterator.get_next()
```

**Saving iterator state** - `tf.contrib.data.make_saveable_from_iterator()` function creates a Saveable object  from an iterator to save and restore the current state of the iterator or the whole input pipeline.  this will be added to `tf.train.Saver` variables list or the `tf.GraphKeys.SAVEABLE_OBJECTS` collection for saving  and restoring  in the manner of a `tf.Variable` eg:-

```python
saveable = tf.contrib.data.make_saveable_from_iterator(iterator)

# save the iterator state by adding it to the saveable objects collection.
tf.add_to_collection(tf.GraphKeys.SAVABLE_OBJECTS, saveable)
saver = tf.train.Saver()

with tf.Session() as sess:    
    if should_checkpoint:
        saver.save(path_to_checkpoint)

# restore  the iterator state.
with tf.Session() as sess:
    saver.restore(sess, path_to_checkpoint)        
```

#### Reading Input Data

**Numpy array**  - if data fits in memory then simplest way is to convert it into `tf.Tensor` objects and use the `Dataset.from_tensor_slices()`

```python
with np.load("path/to/data.npy") as data :
    features = data['features']
    labels = data['labels']

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
```

**TFRecords Format** - TFRecord file format is a simple record-oriented binary format that many TensorFlow applications use for training data. The `tf.data.TFRecordDataset` class enables us to stream over the contents of one or more TFRecord files as part of an input pipeline.

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
```

and this is how we could make use of the `tf.placeholder` for dynamically working with the TFRecords.

```python
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

# Initialize `iterator` with training data.
training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

# Initialize `iterator` with validation data.
validation_filenames = ["/var/data/validation1.tfrecord", ...]
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
```

**Working with Text Data**

The `tf.data.TextLineDataset` provides an easy way to extract lines from one or more text files

```python
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
dataset = tf.data.TextLineDataset(filenames)
```

we could also make use of the `Dataset.filter()`, `Dataset.skip` functions along with the `Dataset.flat_map()` for data processing. 

```python
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]

dataset = tf.data.Dataset.from_tensor_slices(filenames)

# Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
# and then concatenate their contents sequentially into a single "flat" dataset.
# * Skip the first line (header row).
# * Filter out lines beginning with "#" (comments).
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        .skip(1)
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))
```

**Working with CSV Data**

` tf.contrib.data.CsvDataset()` class provides a way to extract records from one or more csv files. the `CsvDataset` will provide us with tuple of elements. `CsvDataset` also accepts filenames as a `tf.tensor()` and hence can be used like other functions described above.

```python
filenames = ["/var/data/file1.csv", "/var/data/file2.csv"]
record_defaults = [tf.float32] * 8   # Eight required float columns
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)
```

we can also select certain columns,  remove certain rows etc in csv as per requirement.

```python
# Creates a dataset that reads all of the records from two CSV files with
# headers, extracting float data from columns 2 and 4.
record_defaults = [[0.0]] * 2  # Only provide defaults for the selected columns
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[2,4]
```

#### Preprocessing Data with Dataset.map()

The `Dataset.map(f)` transformation produces a new dataset by applying a given function `f` to each element of the input dataset. It is based on the `map()` function that is commonly applied to lists (and other structures) in functional programming languages.  lets look into an example.

```python
def _parse_function(example_proto):
  features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["image"], parsed_features["label"]

filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
```

#### Preprocessing Images with tf.Dataset.map()

below example let us give an understanding on how to work with images in tf.Dataset

```python
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# A vector of filenames.
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
```

**Applying Python Logic in Map**

```python
import cv2

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
labels = [0, 37, 29, 1, ...]

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype])))
dataset = dataset.map(_resize_function)
```



### Batching Dataset Elements

**Simple Batching**

The simplest form of batching stacks `n` consecutive elements of a dataset into a single element. The `Dataset.batch()` transformation does exactly this, with the same constraints as the `tf.stack()` operator, applied to each component of the elements:	

```python
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
```

**Batching with padding**

sometimes we may need to including padding in our dataset, esp when the dataset we have is not of fixed length.

```python
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                               #      [5, 5, 5, 5, 5, 0, 0],
                               #      [6, 6, 6, 6, 6, 6, 0],
                               #      [7, 7, 7, 7, 7, 7, 7]]
```

### Training Workflows

**Processing multiple epochs **- simplest way to iterate over a dataset in multiple epochs is to use the `Dataset.repeat()` transformation

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.repeat(10)
dataset = dataset.batch(32)
```

sometime we want to receive a signal at the end of each epoch, we can write  a training loop  that catches the `tf.errors.OutOfRangeError` at the end of dataset

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Compute for 100 epochs.
for _ in range(100):
  sess.run(iterator.initializer)
  while True:
    try:
      sess.run(next_element)
    except tf.errors.OutOfRangeError:
      break

  # [Perform end-of-epoch calculations here.]
```

**Random Shuffling  of input data**

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

