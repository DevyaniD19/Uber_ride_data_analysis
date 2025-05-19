import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv(r"C:\Users\ddeore\OneDrive - SyBridge Technologies\Documents\DD\Python\Uber_ride_data_analysis\UberDataset.csv")

print("\nFirst 5 rows of the database are:\n", dataset.head())
print("\nN(rows,columns):\n", dataset.shape)
print("\nDatabase Info:\n")
dataset.info()

print("\nBefore cleaning CATEGORY counts:\n", dataset['CATEGORY'].value_counts(dropna=False))
print("\nBefore cleaning PURPOSE counts:\n", dataset['PURPOSE'].value_counts(dropna=False))

# Data Preprocessing
dataset['PURPOSE'] = dataset['PURPOSE'].fillna("NOT")
dataset['START_DATE'] = pd.to_datetime(dataset['START_DATE'], errors='coerce')
dataset['END_DATE'] = pd.to_datetime(dataset['END_DATE'], errors='coerce')

dataset['date'] = dataset['START_DATE'].dt.date
dataset['time'] = dataset['START_DATE'].dt.hour

dataset['day-night'] = pd.cut(
    x=dataset['time'],
    bins=[-1, 10, 15, 19, 24],
    labels=['Morning', 'Afternoon', 'Evening', 'Night']
)

# Drop rows only where important datetime fields failed
dataset.dropna(subset=['START_DATE', 'END_DATE'], inplace=True)

dataset.drop_duplicates(inplace=True)

print("\nAfter cleaning CATEGORY counts:\n", dataset['CATEGORY'].value_counts(dropna=False))
print("\nAfter cleaning PURPOSE counts:\n", dataset['PURPOSE'].value_counts(dropna=False))

# Object columns
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)

unique_values = {}
for col in object_cols:
    unique_values[col] = dataset[col].nunique()

print("\nObject columns are:\n", object_cols)
print("\nUnique values in object columns:\n", unique_values)

# Data Visualization
#1) 
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=dataset['CATEGORY'])
plt.xticks(rotation=90)
plt.subplot(1, 2, 2)
sns.countplot(x=dataset['PURPOSE'], order=dataset['PURPOSE'].value_counts().index)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
#2)
sns.countplot(dataset['day-night'])
plt.xticks(rotation=90)
plt.show()
#3)
plt.figure(figsize=(15, 5))
sns.countplot(data=dataset, x='PURPOSE', hue='CATEGORY')
plt.xticks(rotation=90)
plt.show()

#OneHotEncoder to categories Category and Purpose
object_cols = ['CATEGORY', 'PURPOSE']
OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(dataset[object_cols]))
OH_cols.index = dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = dataset.drop(object_cols, axis=1)
dataset = pd.concat([df_final, OH_cols], axis=1)

#correlation -> Heatmap
# correlation -> Heatmap

numeric_dataset = dataset.select_dtypes(include=['number'])

plt.figure(figsize=(10,6))

sns.heatmap(
    numeric_dataset.corr(),
    cmap='BrBG',
    fmt='.2f',
    linewidths=2,
    annot=True
)

plt.title("Correlation Heatmap")
plt.show()

#month by month data analysis
dataset['MONTH'] = pd.DatetimeIndex(dataset['START_DATE']).month
month_label = {1.0: 'Jan', 2.0: 'Feb', 3.0: 'Mar', 4.0: 'April',
               5.0: 'May', 6.0: 'June', 7.0: 'July', 8.0: 'Aug',
               9.0: 'Sep', 10.0: 'Oct', 11.0: 'Nov', 12.0: 'Dec'}
dataset["MONTH"] = dataset.MONTH.map(month_label)

mon = dataset.MONTH.value_counts(sort=False)
df = pd.DataFrame({"MONTHS": mon.values,
                   "VALUE COUNT": dataset.groupby('MONTH',
                                                  sort=False)['MILES'].max()})

p = sns.lineplot(data=df)
p.set(xlabel="MONTHS", ylabel="VALUE COUNT")
plt.show()

#Visualization for days data
dataset['DAY'] = dataset.START_DATE.dt.weekday
day_label = {
    0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thus', 4: 'Fri', 5: 'Sat', 6: 'Sun'
}
dataset['DAY'] = dataset['DAY'].map(day_label)
day_label = dataset.DAY.value_counts()
sns.barplot(x=day_label.index, y=day_label);
plt.xlabel('DAY')
plt.ylabel('COUNT')
plt.show()

#visualize miles column
sns.distplot(dataset[dataset['MILES']<40]['MILES'])
plt.show()

