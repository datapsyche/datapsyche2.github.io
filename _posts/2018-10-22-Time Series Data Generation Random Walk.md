---
layout: post
title: Time Series Generation -An exploratory Study
categories: DataGeneration, RandomWalk, TimeSeries
---
​	So a simple google search on above topic gave quite a lot of results, along with some research papers and links to custom libraries. I was trying to understand the varies methodologies that currently exist to generate dummy data based on a say a very small number of samples. This could be very mush relatable for quite a lot of data scientist.  It is a very common occurance that your stake holder might just say "This is all the data i have, now show us what you could come with this". Well my study here is to come up with an effective solution for all the data scientist who are challenged by their stake holders.

### Random Walk (A better approach than Random Generation)

​	So the first thing that crossed my mind was create a list of values and then use random function and pick random values from the list to extrapolate the data. Well this could work, but that would be naive imagine the situation where we would love the generated data to follow the same trend of the sample data. Random Walk is one such method that could come to our help. 

**What is random walk?**

​	So my Primary understanding about Random Walk was that we are choosing a random number from a given random set / range of values. But I was clearly wrong.!!

[Jason Brownlee's Blog on Random Walk](https://machinelearningmastery.com/gentle-introduction-random-walk-times-series-forecasting-python/)  corrected my basic understanding of the concept of Random Walk in Time Series domain. The most important factor one needs to have about **random walk** is that *"the next value in the sequence is a modification of the previous value in the sequence."* Or n my words the next time step is *not random* it is *dependent on previous value*

This could be mathematicaly explained as  `y(t) = B0 + B1*X(t-1) + e(t)` where  `y(t)` is the next value in the series, `B0`  is the constant drift / trend to be associate with `y(t)`, `B1` is the weight of effect of previous time step (this is value is generally between -1 & 1) , `e(t)` is the stochastic error to be induced the time series. So, We have a perfect Equation into which we need to feed the input values to get a **Random Walk** effect. Now let us check out how to implement this in python.

```(python) 
from random import seed
import random
from matplotlib import pyplot
seed(120)
B0=0.002
random_walk = list()
random_walk.append(B0-1 if random.random() < 0.5 else B0+1)
for i in range(1, 1000):
	movement = -1 if random.random() < 0.5 else 1
	value = B0+random_walk[i-1] + movement+random.randint(-100,100)*.001
	random_walk.append(value)
pyplot.plot(random_walk)
pyplot.show()
```

![output](/home/jithin/github/datapsyche.github.io/public/RandomWalk.png)

So this looks a time series data, and generated randomly using random walk methodology. This could be used to comeup with a random dataset which would actualy resembles a realworld random process (like stock market). 

Reference : [Jason Brownlee's Blog on Random Walk](https://machinelearningmastery.com/gentle-introduction-random-walk-times-series-forecasting-python/)  