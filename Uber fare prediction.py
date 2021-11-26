#!/usr/bin/env python
# coding: utf-8

# train.csv - Input features and target fare_amount values for the training set (about 55M rows).
# test.csv - Input features for the test set (about 10K rows). Your goal is to predict fare_amount for each row.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


train_df=pd.read_csv(r"train.csv",nrows=200)
test_df=pd.read_csv(r"test.csv")
print (train_df.shape)
print (train_df.columns)
print (test_df.shape)
print (test_df.columns)


# In[3]:


train_df.info()


# #here we can see there are 8columns in which 6 numerics and 2 are object.
# #Lets change the type of pickup_datetime from object to DateTime

# In[4]:


train_df["pickup_datetime"]=pd.to_datetime(train_df['pickup_datetime'])


# In[5]:


train_df.head()


# #As this is Taxi fare data and we know there are many factors which affect the price of taxi like 
# 1. Travelled distance
# 2. Time of Travel
# 3. Demand and Availability of Taxi
# 4. Some special places are more costlier like Airport or other places where there might be toll

# In[6]:


#Lets see the statisitics of our data


# In[7]:


train_df.describe()


# #Here first thing which we can see is minimum value of fare is negative which is -62 which is not the valid value, so we need to remove the fare which are negative values.
# #Secondly, passenger_count minimum value is 0 and maximum value is 208 which impossible, so we need to remove them as well, for safer side we can think that a taxi can have maximum 7 people.

# In[8]:


#Lets check if there is any null value
train_df.isnull().sum()


# #Here we can see there are 14 null values in drop_off latitude and longitude. as removing 14 to 28 rows from our huge dataset will not affect our analysis so, lets remove the rows having null values 
# 

# In[9]:


train_df.dropna(inplace=True)
print(train_df.isnull().sum())


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


sns.distplot(train_df['fare_amount'])


# #In distribution plot also it can be seen that there are some values which are negative fare

# In[12]:


sns.distplot(train_df['pickup_latitude'])


# #Here we can see minimum value is going to be less than even -3000 which is not correct value and also on positive side also going more than 2000

# In[13]:


sns.distplot(train_df['pickup_longitude'])


# #Here also negative and positive values are excedding far behond the real limit.

# In[14]:


sns.distplot(train_df['dropoff_longitude'])


# In[15]:


#Similarly here also same issue


# In[16]:


sns.distplot(train_df['dropoff_latitude'])


# In[17]:


#here also we have noisy data as given value of dropoff_latitude and longitude are excedding


# In[18]:


#lets look min and max value in test dataset of latitude and longitude


# In[19]:


print("drop_off latitude min value",test_df["dropoff_latitude"].min())
print("drop_off latitude max value",test_df["dropoff_latitude"].max())
print("drop_off longitude min value", test_df["dropoff_longitude"].min())
print("drop_off longitude max value",test_df["dropoff_longitude"].max())
print("pickup latitude min value",test_df["pickup_latitude"].min())
print("pickup latitude max value",test_df["pickup_latitude"].max())
print("pickup longitude min value",test_df["pickup_longitude"].min())
print("pickup longitude max value",test_df["pickup_longitude"].max())


# #we can see what is range of latitude and longitude of our test dataset, lets keep the range same in our train set so that even noisy data is remove and we have only the values which belongs to new york

# In[20]:


min_longitude=-74.263242,
min_latitude=40.573143,
max_longitude=-72.986532, 
max_latitude=41.709555


# In[21]:


#lets drop all the values which are not coming in above boundary, as those are noisy data


# In[22]:


tempdf=train_df[(train_df["dropoff_latitude"]<min_latitude) | (train_df["pickup_latitude"]<min_latitude) | (train_df["dropoff_longitude"]<min_longitude) | (train_df["pickup_longitude"]<min_longitude) | (train_df["dropoff_latitude"]>max_latitude) | (train_df["pickup_latitude"]>max_latitude) | (train_df["dropoff_longitude"]>max_longitude) | (train_df["pickup_longitude"]>max_longitude) ]
print("before droping",train_df.shape)
train_df.drop(tempdf.index,inplace=True)
print("after droping",train_df.shape)


# In[23]:


#lets remove all those rows where fare amount is negative


# In[24]:


print("before droping", train_df.shape)
train_df=train_df[train_df['fare_amount']>0]
print("after droping", train_df.shape)


# #On different day and time there would be different price like during eveing price would be more compare to afternoon, during christmas price would be different and similarly on weekends price would be different compare to week days. so lets create some extra features which will take care of all these things

# In[25]:


import calendar
train_df['day']=train_df['pickup_datetime'].apply(lambda x:x.day)
train_df['hour']=train_df['pickup_datetime'].apply(lambda x:x.hour)
train_df['weekday']=train_df['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])
train_df['month']=train_df['pickup_datetime'].apply(lambda x:x.month)
train_df['year']=train_df['pickup_datetime'].apply(lambda x:x.year)


# In[26]:


train_df.head()


# In[27]:


#here we can see that week are in monday , tuesday and so on. So we need convert them in numerical for


# In[28]:


train_df.weekday = train_df.weekday.map({'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6})


# In[29]:


train_df.info()


# In[30]:


# we will keep only those rows where number of passangers are less than or equal to 8


# In[31]:


train_df=train_df[train_df['passenger_count']<=8]


# In[32]:


train_df.info()


# In[33]:


#here key column and pickup_datetime columns are not needed as we have already created variables extracted from it


# In[34]:


train_df.drop(["key","pickup_datetime"], axis=1, inplace=True)


# In[35]:


train_df.info()


# #lets divide the data set into train and validation test set

# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


x=train_df.drop("fare_amount", axis=1)


# In[38]:


y=train_df['fare_amount']


# In[39]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)


# In[40]:


x_train.head()


# In[41]:


x_test.head()


# In[42]:


x_train.shape


# In[43]:


x_test.shape


# In[44]:


#Lets run the model.
#As we have to build regression model, lets start with linear regression model


# In[45]:


from sklearn.linear_model import LinearRegression


# In[46]:


lrmodel=LinearRegression()
lrmodel.fit(x_train, y_train)


# In[47]:


predictedvalues = lrmodel.predict(x_test)


# In[48]:


#lets calculate rmse for linear Regression model
from sklearn.metrics import mean_squared_error
lrmodelrmse = np.sqrt(mean_squared_error(predictedvalues, y_test))
print("RMSE value for Linear regression is", lrmodelrmse)


# In[49]:


#Lets see with Random Forest and calculate its rmse
from sklearn.ensemble import RandomForestRegressor
rfrmodel = RandomForestRegressor(n_estimators=100, random_state=101)


# In[50]:


rfrmodel.fit(x_train,y_train)
rfrmodel_pred= rfrmodel.predict(x_test)


# In[51]:


rfrmodel_rmse=np.sqrt(mean_squared_error(rfrmodel_pred, y_test))
print("RMSE value for Random forest regression is ",rfrmodel_rmse)


# In[52]:


#RandomForest Regressor is giving good value, so we can use it as final model

