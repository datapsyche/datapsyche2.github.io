So this is the last chapter of  walk through of the concepts that i have tried to learn in past 5 days about tensorflow. There are 4 highlevel API's that were introduced in the latest version of TensorFlow (1.9 as on 10-31-2018). lets discuss about third of those 4 high level API's, we have already coverd tf.keras, EagerExecution and Importing Data previously. lets look into Estimators perhaps the most important one.!

### Estimators

`tf.estimator` is a highlevel TensorFlow API that is meant to simplfy machine learning programming. `estimator` encapsulate the following actions, **Training, Evaluation, Prediction, Export for serving** .We can either use the pre-made estimators or we can write custom estimators based on the `tf.estimator.Estimator` class.



#### Summary of white paper published by Google.

*Why we need estimators ?* - The paper discuss about the natural tension between flexibility and simplicity with robustness. DSL based model architecture (Domain Specific Language) helps in making coming up with machine learning models that are simple and highly robust however with very less flexibility. DSL systems are hard to maintain as the research is still advancing in a rapid pace. This lead to the development of **TensorFlow**. 

Estimator framework described in this paper is implemented on top of tensorflow which provides an interface similar to Scikit-learn with some adaptation for productionization.  Preference for functions and closures over objects. callbacks are common in closures. Layer functions are tensor in tensor out operations. 

**Various Components of Estimator**

*Layers*  - A layer is simply a reusable part of code, and can be as simple as a fully connected neural network layer.layers are implemented as free functions, taking Tensors as input arguments and returning Tensors.

*Estimator* -The interface for users of Estimator is loosely modeled afer Scikit-learn and consists of only four methods: train - trains the model, given training data, evaluate - computes evaluation metrics over test data, predict- performs inference on new data given a trained model, and export_savedmodel - exports a SavedModel. Estimator hides the tensorflow concepts like Graph and session from the user. 

*Canned Estimator* - There are many model architectures commonly used by researchers and practitioners. We decided to provide those  architectures as canned Estimators so that users don’t need to rewrite the same models again and again. 



