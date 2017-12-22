---
layout: post
title:  "Building custom layers in Keras"
date:   2017-11-01 01:20:33 +0530
categories: Tutorials
description : Tutorial on creating custom layers in Keras
---
_Originally published at [Saama website](https://www.saama.com/blog/deep-learning-diaries-building-custom-layers-in-keras/)_
## About Keras
Keras is currently one of the most commonly used deep learning libraries today. And part of the reason why it's so popular is its API. Keras was built as a high-level API for other deep learning libraries ie Keras as such does not perform low-level tensor operations, instead provides an interface to its backend which are built for such operations. This allows Keras to abstract a lot of the underlying details and allows the programmer to concentrate on the architecture of the model. Currently Keras supports Tensorflow, Theano and CNTK as its backends.

Let's see what I mean. Tensorflow is one of the backends used by Keras. Here's the code for MNIST classification in [TensorFlow](https://www.github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_deep.py) and [Keras](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py). Both models are nearly identical and applies to the same problem. But if you compare the codes you get an idea of the abstraction Keras provides you with. The entire model is defined within 10 lines of code !

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
```

You can visit the [official documentation]((https://keras.io/)) for understanding the basic usage of Keras.

### Sequential API vs Functional API
Keras has two different APIs, [Sequential](https://keras.io/getting-started/sequential-model-guide/) and [Functional](https://keras.io/getting-started/functional-api-guide/).

The sequential model is helpful when your model is simply one layer after the other. You can use `model.add()` to stack layers and `model.compile` to compile the model with required loss function and optimizers. The example at the beginning uses the sequential model. As you can see, the sequential model is simple in its usage.

The Keras functional API brings out the real power of Keras. If you want to build complex models with multiple inputs or models with shared layers, functional API is the way to go. Let's see the example from the docs

```python
from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # starts training
```

Here, the layers take a more functional form compared to the sequential model. The inputs to each layer are explictly specified and you have access to the output of each layer. This allows you to share the tensors with multiple layers. The functional API also gives you control over the model inputs and outputs as seen above.

### Keras computational graph
Before we write our custom layers let's take a closer look at the internals of Keras computational graph. Keras has its own graph which is different from that of it's underlying backend. The Keras topology has 3 key classes that is worth understanding
- `Layer` encapsules the weights and the associated computations of the layer. The `call` method of a layer class contains the layer's logic. The layer has `inbound_nodes` and `outbound_nodes` attributes. Each time a layer is connected to some new input, a node is added to `inbound_nodes`. Each time the output of a layer is used by another layer, a node is added to `outbound_nodes`. The layer also carries a list of trainable and non-trainable weights of the layer.
- `Node` represents the connection between two layers. `node.outbound_layer` points to the layer that converts the input tensor into output tensor. `node.inbound_layers` is a list of layers from where the input tensors originate. The node object carries other information like input and output shapes, masks etc along with the actual input tensors and output tensors.
- `Container` is a directed acyclic graph of layers connected using nodes. It represents the topology of the model. This graph ensures the correct propagation of gradients to the inputs. The actual model couples the optimzer and training routines along with this.


## Custom layers
Despite the wide variety of layers provided by Keras, it is sometimes useful to create your own layers like when you need are trying to implement a new layer architecture or a layer that doesn't exist in Keras. Custom layers allow you to set up your own transformations and weights for a layer. Remember that if you do not need new weights and require stateless transformations you can use the [Lambda](https://keras.io/layers/core/#lambda) layer.

Now let’s see how we can define our custom layers. As of Keras 2.0 there are three functions that needs to be defined for a layer
- `build(input_shape)`
- `call(input)`
- `compute_output_shape(input_shape)`

The `build` method is called when the model containing the layer is built. This is where you set up the weights of the layer. The `input_shape` is accepted as an argument to the function.

The `call` method defines the computations performed on the input. The function accepts the input tensor as its argument and returns the output tensor after applying the required operations.

Finally, we need to define the `compute_output_shape` function that is required for Keras to infer the shape of the output. This allows Keras to do shape inference without actually executing the computation. The `input_shape` is passed as the argument.

### Example
Now lets build our custom layer. For the sake of simplicity we'll be building a vanilla fully-connected layer (called `Dense` in Keras). First let's make the required imports

```python
from keras import backend as K
from keras.engine.topology import Layer
```

Now let's create our layer named `MyDense`. Our new layer must inherit from the base `Layer` class. Set up `__init__` function to accept the number of units (accepted as output_dim) in the fully connected layer.

```python
class MyDense(Layer):

    def __init__(self, output_dim, **kwargs):
        self.units = output_dim
        super(MyLayer, self).__init__(**kwargs)
```

The `build` function is where we define the weights of the layer. The weights can be instantiated using the `add_weight` method of the layer class. The `name` and `shape` arguments determine the name used for the backend variable and the shape of the weight variable respectively. If your input is of the shape `(batch_size, input_dim)` your weights need to be of the shape `(input_dim, output_dim)`. Here output dimension denotes the number of units in the layer.

Optionally you may also specify an [initializer](https://keras.io/initializers/), [regularizer](https://keras.io/regularizers/) and a [constraint](https://keras.io/constraints/). The `trainable` argument can be set to False to prevent the weight from contributing to the gradient. Make sure you also call the `build` method of the base layer class.

```python
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)
```

The `call` function houses the logic of the layer. For our fully connected layer it means that we have to calculate the dot product between the weights and the input. The input is passed as a parameter and the result of the dot product is returned.
```python
    def call(self, x):
        y = K.dot(x, self.kernel)
        return y
```

The `compute_output_shape` is a helper function to specify the change in shape of the input when it passes through the layer. Here our output shape will be `(batch_size, output_dim)`
```python
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

That's it, you're good to go. You can use `MyDense` layer just like any other layer in Keras.

## Keras vs Other DL Frameworks
I've seen a lot of discussions comparing deep learning frameworks including Keras and personally I think Keras should not be on the list. As I mentioned earlier, Keras is technically not a deep learning framework, it's an API. It runs on top of other DL frameworks. The power of Keras is in it's abstraction while still giving you sufficient control over your architecture.

But in case you decide that you need to play with lower level tensor operations you can always go for other frameworks. Tensorflow seems to be the most popular these days. Backed by Google, it even has C++ and Go APIs. Theano was one of the first DL frameworks, but have been discontinued recently. PyTorch is an interesting competitor with it's dynamic graphs. Unlike Tensorflow or Theano, which has static graphs, the computational graphs in [PyTorch](http://pytorch.org/about/) are built dynamically for each input. It is becoming increasingly popular amongst researchers due to its flexibility. 

If you ask me Keras is sufficient for most purposes. Even when you experiment with new architectures the functional API combined with the access to backend functions can get the job done often. But that's just me, a lot of people directly go for Tensorflow and other libraries.