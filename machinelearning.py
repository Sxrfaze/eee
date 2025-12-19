from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
files = 'sustainable_waste_management_dataset_2024.csv'
df = pd.read_csv(files, parse_dates=['date'])

df.head()

selected_features = ['area', 'population', 'waste_kg', 'recyclable_kg', 'organic_kg', 'overflow', 'is_weekend', 'is_holiday', 'recycling_campaign','temp_c','rain_mm']
X = df[selected_features]
y = df['collection_capacity_kg']


X = pd.get_dummies(X, columns=['area'], drop_first=True)

df_combined = pd.concat([X, y], axis=1)
df_combined.dropna(inplace=True)

X = df_combined.drop(columns=['collection_capacity_kg'])
y = df_combined['collection_capacity_kg']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("MSE: ", mean_squared_error(Y_test, Y_pred))
print("R squared: ", r2_score(Y_test, Y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual Collection Capacity')
plt.ylabel('Predicted Collection Capacity')
plt.title('Predicted vs. Actual Collection Capacity')
plt.legend()
plt.grid(True)
plt.show()