#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression
# 
# 
# Researchers are often interested in setting up a model to analyze the relationship between predictors (i.e., independent variables) and it's corresponsing response (i.e., dependent variable). Linear regression is commonly used when the response variable is continuous.  One assumption of linear models is that the residual errors follow a normal distribution. This assumption fails when the response variable is categorical, so an ordinary linear model is not appropriate. This newsletter presents a regression model for response variable that is dichotomous–having two categories. Examples are common: whether a plant lives or dies, whether a survey respondent agrees or disagrees with a statement, or whether an at-risk child graduates or drops out from high school.
# 
# In ordinary linear regression, the response variable (Y) is a linear function of the coefficients (B0, B1, etc.) that correspond to the predictor variables (X1, X2, etc.,). A typical model would look like:
# 
#     Y = B0 + B1*X1 + B2*X2 + B3*X3 + … + E
# 
# For a dichotomous response variable, we could set up a similar linear model to predict individual category memberships if numerical values are used to represent the two categories. Arbitrary values of 1 and 0 are chosen for mathematical convenience. Using the first example, we would assign Y = 1 if a plant lives and Y = 0 if a plant dies.
# 
# This linear model does not work well for a few reasons. First, the response values, 0 and 1, are arbitrary, so modeling the actual values of Y is not exactly of interest. Second, it is the probability that each individual in the population responds with 0 or 1 that we are interested in modeling. For example, we may find that plants with a high level of a fungal infection (X1) fall into the category “the plant lives” (Y) less often than those plants with low level of infection. Thus, as the level of infection rises, the probability of plant living decreases.
# 
# Thus, we might consider modeling P, the probability, as the response variable. Again, there are problems. Although the general decrease in probability is accompanied by a general increase in infection level, we know that P, like all probabilities, can only fall within the boundaries of 0 and 1. Consequently, it is better to assume that the relationship between X1 and P is sigmoidal (S-shaped), rather than a straight line.
# 
# It is possible, however, to find a linear relationship between X1 and function of P. Although a number of functions work, one of the most useful is the logit function. It is the natural log of the odds that Y is equal to 1, which is simply the ratio of the probability that Y is 1 divided by the probability that Y is 0. The relationship between the logit of P and P itself is sigmoidal in shape. The regression equation that results is:
# 
#     ln[P/(1-P)] = B0 + B1*X1 + B2*X2 + …
# 
# Although the left side of this equation looks intimidating, this way of expressing the probability results in the right side of the equation being linear and looking familiar to us. This helps us understand the meaning of the regression coefficients. The coefficients can easily be transformed so that their interpretation makes sense.
# 
# The logistic regression equation can be extended beyond the case of a dichotomous response variable to the cases of ordered categories and polytymous categories (more than two categories).

# # Mathematics behind Logistic Regression

# ## Notation

# The problem structure is the classic classification problem. Our data set $\mathcal{D}$ is composed of $N$ samples. Each sample is a tuple containing a feature vector and a label. For any sample $n$ the feature vector is a $d+1$ dimensional column vector denoted by ${\bf x}_n$ with $d$ real-valued components known as features. Samples are represented in homogeneous form with the first component equal to $1$: $x_0=1$. Vectors are bold-faced. The associated label is denoted $y_n$ and can take only two values: $+1$ or $-1$.
# 
# $$
# \mathcal{D} = \lbrace ({\bf x}_1, y_1), ({\bf x}_2, y_2), ..., ({\bf x}_N, y_N) \rbrace \\
# {\bf x}_n = \begin{bmatrix} 1 & x_1 & ... & x_d \end{bmatrix}^T 
# $$

# ## Learning Algorithm

# The learning algorithm is how we search the set of possible hypotheses (hypothesis space $\mathcal{H}$) for the best parameterization (in this case the weight vector ${\bf w}$). This search is an optimization problem looking for the hypothesis that optimizes an error measure.

# There is no sophisticted, closed-form solution like least-squares linear, so we will use gradient descent instead. Specifically we will use batch gradient descent which calculates the gradient from all data points in the data set.

# Luckily, our "cross-entropy" error measure is convex so there is only one minimum. Thus the minimum we arrive at is the global minimum.

# Gradient descent is a general method and requires twice differentiability for smoothness. It updates the parameters using a first-order approximation of the error surface.
# 
# $$
# {\bf w}_{i+1} = {\bf w}_i + \nabla E_\text{in}({\bf w}_i)
# $$

# To learn we're going to minimize the following error measure using batch gradient descent.
# 
# $$
# e(h({\bf x}_n), y_n) = \ln \left( 1+e^{-y_n \; {\bf w}^T {\bf x}_n} \right) \\
# E_\text{in}({\bf w}) = \frac{1}{N} \sum_{n=1}^{N} e(h({\bf x}_n), y_n) = \frac{1}{N} \sum_{n=1}^{N} \ln \left( 1+e^{-y_n \; {\bf w}^T {\bf x}_n} \right)
# $$

# We'll need the derivative of the point loss function and possibly some abuse of notation.
# 
# $$
# \frac{d}{d{\bf w}} e(h({\bf x}_n), y_n)
# = \frac{-y_n \; {\bf x}_n \; e^{-y_n {\bf w}^T {\bf x}_n}}{1 + e^{-y_n {\bf w}^T {\bf x}_n}}
# = -\frac{y_n \; {\bf x}_n}{1 + e^{y_n {\bf w}^T {\bf x}_n}}
# $$

# With the point loss derivative we can determine the gradient of the in-sample error:
# 
# $$
# \begin{align}
# \nabla E_\text{in}({\bf w})
# &= \frac{d}{d{\bf w}} \left[ \frac{1}{N} \sum_{n=1}^N e(h({\bf x}_n), y_n) \right] \\
# &= \frac{1}{N} \sum_{n=1}^N \frac{d}{d{\bf w}} e(h({\bf x}_n), y_n) \\
# &= \frac{1}{N} \sum_{n=1}^N \left( - \frac{y_n \; {\bf x}_n}{1 + e^{y_n {\bf w}^T {\bf x}_n}} \right) \\
# &= - \frac{1}{N} \sum_{n=1}^N \frac{y_n \; {\bf x}_n}{1 + e^{y_n {\bf w}^T {\bf x}_n}} \\
# \end{align}
# $$

# Our weight update rule per batch gradient descent becomes
# 
# $$
# \begin{align}
# {\bf w}_{i+1} &= {\bf w}_i - \eta \; \nabla E_\text{in}({\bf w}_i) \\
# &= {\bf w}_i - \eta \; \left( - \frac{1}{N} \sum_{n=1}^N \frac{y_n \; {\bf x}_n}{1 + e^{y_n {\bf w}_i^T {\bf x}_n}} \right) \\
# &= {\bf w}_i + \eta \; \left( \frac{1}{N} \sum_{n=1}^N \frac{y_n \; {\bf x}_n}{1 + e^{y_n {\bf w}_i^T {\bf x}_n}} \right) \\
# \end{align}
# $$
# 
# where $\eta$ is our learning rate.

# ### Enough with the theory, now jump to the implimentation. We will look at 2 libraries for the same.

# ## Logistic Regression with statsmodel

# We'll be using the same dataset as UCLA's Logit Regression tutorial to explore logistic regression in Python. Our goal will be to identify the various factors that may influence admission into graduate school.
# 
# The dataset contains several columns which we can use as predictor variables:
# 
#    * gpa
#    * gre score
#    * rank or prestige of an applicant's undergraduate alma mater
#    * The fourth column, admit, is our binary target variable. It indicates whether or not a candidate was admitted our not.

# In[1]:


import numpy as np
import pandas as pd
import pylab as pl
import statsmodels.api as sn


# In[2]:


df = pd.read_csv("binary.csv")
#df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")


# In[3]:


df.head()


# In[5]:


#RENAME THE RANK COLOUMN BECAUSE THERE IS ALSO A DATA FRAME CALLED"RANK"
df.columns = ['admit','gre','gpa','prestige']
df.head()
#df.shape


# ### Summary Statistics & Looking at the data
# Now that we've got everything loaded into Python and named appropriately let's take a look at the data. We can use the pandas function which describes a summarized view of everything. There's also function for calculating the standard deviation, std.
# 
# A feature I really like in pandas is the pivot_table/crosstab aggregations. crosstab makes it really easy to do multidimensional frequency tables. You might want to play around with this to look at different cuts of the data.

# In[6]:


pd.crosstab(df['admit'],df['prestige'],rownames=['admit'])


# In[9]:


df.hist()
pl.show()


# In[8]:





#  ### dummy variables
# pandas gives you a great deal of control over how categorical variables can be represented. We're going dummify the "prestige" column using get_dummies.
# 
# get_dummies creates a new DataFrame with binary indicator variables for each category/option in the column specified. In this case, prestige has four levels: 1, 2, 3 and 4 (1 being most prestigious). When we call get_dummies, we get a dataframe with four columns, each of which describes one of those levels.

# In[10]:


dummy_ranks = pd.get_dummies(df["prestige"],prefix="prestige")


# In[14]:


dummy_ranks.head()


# In[16]:


# CREATING A CLEAN DATA FRAME
cols_to_keep = ["admit","gre","gpa"]
data = df[cols_to_keep].join(dummy_ranks.loc[:,"prestige_2":])
data.head()


# Once that's done, we merge the new dummy columns with the original dataset and get rid of the prestige column which we no longer need.
# 
# Lastly we're going to add a constant term for our logistic regression. The statsmodels function we would use requires intercepts/constants to be specified explicitly.
# 
# ### Performing the regression
# Actually doing the logistic regression is quite simple. Specify the column containing the variable you're trying to predict followed by the columns that the model should use to make the prediction.
# 
# In our case we'll be predicting the admit column using gre, gpa, and the prestige dummy variables prestige_2, prestige_3 and prestige_4. We're going to treat prestige_1 as our baseline and exclude it from our fit. This is done to prevent multicollinearity, or the dummy variable trap caused by including a dummy variable for every single category.

# In[17]:


#ADDING THE INTERCEPT MANUALLY
data["intercept"] = 1.0
data.head()


# In[19]:


train_cols = data.columns[1:]

logit = sn.Logit(data["admit"],data[train_cols])


# In[20]:


results = logit.fit()


# Since we're doing a logistic regression, we're going to use the statsmodels Logit function. For details on other models available in statsmodels, check out their docs here.
# 
# ### Interpreting the results
# One of my favorite parts about statsmodels is the summary output it gives. If you're coming from R, I think you'll like the output and find it very familiar too.

# In[26]:


ironman = results.predict([800,4,0,0,0,1.0])


# In[29]:


print(ironman)


# In[28]:


results.summary()


# In[ ]:




