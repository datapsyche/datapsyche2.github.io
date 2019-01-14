## Data Mining Map by Saeed Sayad

**Data Science** can be broadly divided into two approaches *explaining the past*  or *predicting the future* by means of data analysis. Data Science is a multi disciplinary field that combines statistics, machine learning, artificial intelligence and database technology.

![](/home/jithin/github/datapsyche.github.io/img/2019-11-11-datascienceMap.png)

Business have accumulated data over the years and with the help of data science we are able extract valuable knowledge from this data. Lets understand how each field in above diagram contribute to Data Science.

**Statistics** is used in Data Science for collecting , classifying, summarising, organising, analysing and interpreting data. **Artificial Intelligence** contributes to Data Science by simulating intelligent behaviours from the underlying data. **Machine Learning** contributes to data science by coming up with algorithms that improve automatically through experience. **Database Technology** is necessary for collecting, storing and managing data so users can retrieve, add, update or remove data.

##### Now what do we do in Data Science ?? 

In Laymans language we analyse data to **explain the past** or to **predict the future**. 



## Explaining the past

For explaining the past we need to do Data Exploration - It is all about describing the data by means of statistical and visualization technique. Data Exploration helps in order to bring important aspects of data into focus for further analysis.

* Univariate Analysis -  Exploring variables one by one . Variables can be of two types categorical or numerical. Each type of variable has is own recommended way for analysis or for graphical plotting.

  * Categorical Variables - A categorical or discrete variable has two or more categories (values). There are two types of categorical variables

    * Nominal Variable -  No intrinsic ordering for its categories (eg - Gender)
    * Ordinal Variable - A clear ordering is there (eg - Temperature [low, medium, high]

    **Frequency tables** are the best way to analyse such variables 

    **Pie Chart** and **Bar Chart** are commonly used for visual analysis.

    ![](/home/jithin/github/datapsyche.github.io/img/Univariate.png)

  * **Numerical Variables** - takes any value within a finite  or infinite interval eg:- *height, weight, temperature, blood glucose*. There are two types of numerical variables, intervals and ratio. 

    * **Interval Variables** - Values whose differences are interpret able. (*temperature in centigrade*). Data in Interval scale can be added or subtracted but cannot be meaningfully multiplied or divided.
    * **Ratio Variable** - Data in ratio variable has values with a true zero and can be added, subtracted ,multiplied or divided. (eg- *weight*)

    How do we analyse Numerical variables ? Below table describe the various methods to analyse Numerical variable

    | Statistics               | Visualization | Equation                                                     | Description                                                  |
    | ------------------------ | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | Count                    | Histogram     | *N*                                                          | Number of observations                                       |
    | Minimum                  | Box Plot      | *Min*                                                        | smallest value among the observations                        |
    | Maximum                  | Box Plot      | *Max*                                                        | largest value among the observations                         |
    | Mean                     | Box Plot      | ![](/home/jithin/github/datapsyche.github.io/img/mean.png)   | Sum of values / count                                        |
    | Median                   | Box Plot      | ![](/home/jithin/github/datapsyche.github.io/img/Median.png) | Middle value below and above lies equal number of values     |
    | Mode                     | Histogram     |                                                              | Most frequent value in observation set                       |
    | Quantile                 | Box Plot      | *Q~k~*                                                       | Cut points that divides observations into multiple groups with equal number of values |
    | Range                    | Box plot      | *Max - Min*                                                  | Difference between Maximum and minimum                       |
    | Variance                 | Histogram     | ![](/home/jithin/github/datapsyche.github.io/img/Variance.png) | Measure of Data Dispersion                                   |
    | Standard Deviation       | Histogram     | ![](/home/jithin/github/datapsyche.github.io/img/StDev.png)  | Square root of Variance                                      |
    | Coefficient Of Deviation | Histogram     | ![](/home/jithin/github/datapsyche.github.io/img/CV.png)     | Measure of Data Dispersion divided by Mean                   |
    | Skewness                 | Histogram     | ![](/home/jithin/github/datapsyche.github.io/img/Skewness.png) | Measure of symmetry or asymmetry of data                     |
    | Kurtosis                 | Histogram     |                                                              | Measure of whether the data are peaked or flat relative to a normal distribution |

* **Bivariate Analysis** - Simultaneous analysis of 2 Variables. Explores the concept of relationship between 2 variables. There are 3 types of bivariate analysis

  * **Numerical & Numerical** - 
    * Scatter Plot is a visual representation of two numerical variables, We can infer patterns from this.
    * Linear Correlation - quantifies the linear relationship between two numerical variables
  * **Categorical & Categorical** -
    * Stacked Column Chart - Compares the percentage each category from one variable contributes to a total across categories of second variable 
    * Combination Charts - two or more different type of charts for each variable (bar chart and chart) to show how one variable is affecting other variable
    * Chi Square Test -Used to determine association between categorical variables, based on differences between expected frequencies (e) and observed frequency (n) in one or more categories in the frequency  table. The test returns a probability for the computed chi square and degree of freedom, probability of 0 means complete dependency between categorical variable and probability of 1 means two categorical variables are completely independent
  * **Numerical & Categorical** -
    * Line Chart with error Bars -Error Bars show standard Error in that particular Category.
    * Combination Chart - either line or Bar chart ( line for numerical variable and bar for categorical)
    * Z test and T test - Assess averages of two groups are statistically different from each other If probability between Z is small the difference between two averages is more significant. We use T test when number of observation is less than 30 
    * Anova test Assess whether the averages of more than two groups are statistically different from each other, Analysis is appropriate for comparing the averages of a numerical variable for more than two categories of a categorical variable.



## Predicting the Future

For predicting the future we make use of models. Hence the name Predictive Modeling. Here we try to predict the outcome. if the outcome is Categorical we call it **classification**, if it is numerical we call it **regression**. Descriptive modelling or **clustering** is the assignment of observations into clusters. **Association Rule** can help us find interesting association among observations.



### Classification Algorithms

Here output variable is categorical. Classification algorithms could be broadly divided into 4 main groups.

* Frequency Table Based 

  * Zero R method - simplest classification method exclusively relies on target and ignores all predictors. ZeroR classifier simply predicts the majority class. It has no predictability power however it is usefull for determining Baseline performance as a benchmark for other classification methods.

    ```Construct a frequency  table and select its most frequent value.```

  * One R method - simple yet accurate classifiaction algorithm that generates one rule for each predictor in the data and then selects the rule with smallest total error as its one rule. 
    ```General
    For each predictor,
            For each value of that predictor make a rule as follows:
                    Count how often each value of target appears.
                    Find the most frequent class
                    Make the rule assign that class to this value of the predictor
            Calculate the total error of the rules of each predictor
    Choose the predictor with smallest total error
    ```

  * Naive Bayes Method -  Based on Naive Bayes Theorem with independence assumption between predictors. Naive Bayes model is easy to build, useful for large data set. Naive Bayes often useful and is widely used as it outperforms classification methods.
    $$
    P(c/x) = (P(x/c)*P(c))/P(x)
    $$
    $$ P(c/x) $$ - Posterior Probability 

    $$ P(x/c) $$ - Likelihood

    $$ P(c) $$ - Class probability

    $$ P(x) $$ - Predictor Prior Probability 

    $$ P(c/X)  = P(x~1~/C) * P(x~2~/C) x ....P(x~n~/C) * P(C)$$

    

    

    