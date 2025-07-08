import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')
import os

pwd

df = pd.read_csv('/home/vinod-kumar/Downloads/Shoppers_Behaviour_and_Revenue.csv')
df.head()

df.shape

df.size

df1 = pd.DataFrame({'Info':df.info(),
                    'Isnull':df.isnull().sum(),
                    'nunique':df.nunique(),
                    'unique':df.apply(lambda col:col.unique()),
                   'Sample_Value':df.iloc[7],
                   'dtypes':df.dtypes})
df1

cat = df.select_dtypes(include='object')
num =df.select_dtypes(include='number')

cat.head()

num.head()

print("Revenue conversion rate:", df['Revenue'].mean() * 100, "%")

sns.countplot(data=df, x='VisitorType', hue='Revenue')
plt.title("Revenue by Visitor Type")
plt.show()

monthly_revenue = df.groupby('Month')['Revenue'].mean().sort_index()
monthly_revenue.plot(kind='bar', title='Monthly Revenue Conversion Rate')
plt.ylabel('Conversion Rate')
plt.show()

sns.boxplot(x='Revenue', y='PageValues', data=df)
plt.title("PageValues vs Revenue")
plt.show()

sns.pairplot(df, hue='Revenue', vars=['PageValues', 'Administrative_Duration', 'ProductRelated_Duration'])
plt.show()

plt.figure(figsize=(15,10))
sns.heatmap(num.corr(),annot=True)

## dectecting outliers

for col in num:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f'boxplot of {col}')
    plt.show()
    
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


sns.pairplot(df)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cat:
    df[col] = le.fit_transform(df[col])
    
df.head()

df.dtypes

X = df.drop(columns=('Revenue'))
y = df['Revenue']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestRegressor

refr = RandomForestRegressor()
refr.fit(X_train,y_train)

y_pred = refr.predict(X_test)

print(y_pred)

pip install -U scikit-learn

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print('rmse:',rmse)

import pandas as pd
import matplotlib.pyplot as plt

importances = refr.feature_importances_
features = X.columns

feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
feat_imp.head(10).plot(kind='barh', title='Top 10 Important Features')
plt.gca().invert_yaxis()
plt.show()


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







