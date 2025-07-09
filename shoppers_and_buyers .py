#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')
import os


# In[4]:


import os
os.chdir("/home/vinod-kumar/Downloads")


# In[5]:


df = pd.read_csv('shoppers_and_buyers.csv')
df.head()


# In[37]:


df.shape


# In[4]:


df.size


# In[5]:


df1 = pd.DataFrame({'Info':df.info(),
                    'Isnull':df.isnull().sum(),
                    'nunique':df.nunique(),
                    'unique':df.apply(lambda col:col.unique()),
                   'Sample_Value':df.iloc[7],
                   'dtypes':df.dtypes})
df1


# In[6]:


cat = df.select_dtypes(include='object')
num =df.select_dtypes(include='number')


# In[7]:


cat.head()


# In[8]:


num.head()


# In[9]:


print("Revenue conversion rate:", df['Revenue'].mean() * 100, "%")


# In[10]:


sns.countplot(data=df, x='VisitorType', hue='Revenue')
plt.title("Revenue by Visitor Type")
plt.show()


# ## Returning visitors are more to purchase compared to new_visitors and others

# ## Returning visitors showed more siginifically higher conversion rates ,suggesting loyality is a strong revenue indicator

# In[11]:


monthly_revenue = df.groupby('Month')['Revenue'].mean().sort_index()
monthly_revenue.plot(kind='bar', title='Monthly Revenue Conversion Rate')
plt.ylabel('Conversion Rate')
plt.show()


# November and october month are highly in revenue collecting in the year like with the help of seasons like blackfriday and many more festive seasons

# Month of Feb march and June are not generating revenues as its out of season effect

# In[12]:


sns.boxplot(x='Revenue', y='PageValues', data=df)
plt.title("PageValues vs Revenue")
plt.show()


# In[13]:


sns.kdeplot(data=df, x='PageValues', hue='Revenue', fill=True)
plt.title("Page Value Distribution by Revenue")
plt.show()


# In[14]:


sns.pairplot(df, hue='Revenue', vars=['PageValues', 'Administrative_Duration', 'ProductRelated_Duration'])
plt.show()


# Outliers found in numerical columns like:
# 
# Administrative_Duration
# 
# ProductRelated_Duration
# 
# PageValues

# In[15]:


plt.figure(figsize=(15,10))
sns.heatmap(num.corr(),annot=True)


# PageValues and ProductRelated, ProductRelated_Duration are the most positively correlated with Revenue.
# 
# BounceRates and ExitRates are negatively correlated — users leaving early rarely convert.

# In[16]:


## dectecting outliers

for col in num:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f'boxplot of {col}')
    plt.show()


# In[17]:


## removing outliers

import pandas as pd

# Assuming your DataFrame is named df
# Separate numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Loop through each numerical column and remove outliers using IQR
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Keep only the rows within the IQR range
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


# In[18]:


## removed outliers

print("Data shape after outlier removal:", df.shape)

import seaborn as sns
import matplotlib.pyplot as plt

# Replot boxplots to verify
for col in num_cols:
    plt.figure(figsize=(6, 2))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot after outlier removal: {col}")
    plt.show()


# In[19]:


sns.pairplot(df)


# In[20]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cat:
    df[col] = le.fit_transform(df[col])


# In[21]:


df.head()


# In[22]:


df.dtypes


# In[23]:


X = df.drop(columns=('Revenue'))
y = df['Revenue']


# In[24]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[25]:


from sklearn.ensemble import RandomForestRegressor


# In[26]:


refr = RandomForestRegressor()
refr.fit(X_train,y_train)


# In[27]:


y_pred = refr.predict(X_test)


# In[28]:


print(y_pred)


# In[29]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# In[30]:


print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print('rmse:',rmse)


# In[31]:


import pandas as pd
import matplotlib.pyplot as plt

importances = refr.feature_importances_
features = X.columns

feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
feat_imp.head(10).plot(kind='barh', title='Top 10 Important Features')
plt.gca().invert_yaxis()
plt.show()


# In[32]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor()
}

for name, refr in models.items():
    refr.fit(X_train, y_train)
    preds = refr.predict(X_test)
    print(f"{name} R2: {r2_score(y_test, preds):.3f}")


# | Model                 | R² Score   | Interpretation                                                                                                                             |
# | --------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
# | **Linear Regression** | **-0.002** | Performs slightly better than others, but still fails to capture predictive power. Close to zero, indicating almost no variance explained. |
# | **Gradient Boosting** | **-0.025** | Performs worse than linear regression. Might be overfitting or affected by noise or outliers.                                              |
# | **Random Forest**     | **-0.103** | Performs worst among the three. Likely struggling due to data quality issues or irrelevant/noisy features.                                 |
# 

# In[ ]:




