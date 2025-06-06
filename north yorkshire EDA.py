#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 01:38:51 2025

@author: moeedazam
"""

import os
import pandas as pd

main_folder = '/Users/moeedazam/Downloads/97e2c04cc7b177d386517e0914f1e222c6850b11'
all_dataframes = []

for folder_name in sorted(os.listdir(main_folder)):
    folder_path = os.path.join(main_folder, folder_name)

    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):  # ✅ updated to check for .csv files
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)
                df['Month'] = folder_name
                df['SourceFile'] = file
                all_dataframes.append(df)

# Merge and save
combined_df = pd.concat(all_dataframes, ignore_index=True)
combined_df.to_excel('/Users/moeedazam/Desktop/combined_data_yorkshire.xlsx', index=False)
print("✅ Merged and saved as 'combined_data_yorkshire.xlsx' on Desktop!")

import pandas as pd

# Load the Excel file from your Desktop
df = pd.read_excel("/Users/moeedazam/Desktop/combined_data_yorkshire.xlsx")

# 1. Drop the 'Context' column (completely empty)
if 'Context' in df.columns:
    df.drop(columns=['Context'], inplace=True)

# 2. Drop rows with missing Longitude or Latitude (needed for maps/clustering)
df.dropna(subset=['Longitude', 'Latitude'], inplace=True)

# 3. Fill missing 'Last outcome category' with 'Unknown'
if 'Last outcome category' in df.columns:
    df['Last outcome category'] = df['Last outcome category'].fillna('Unknown')

# 4. Create 'Year' and 'Month_Num' from the 'Month' column
df['Year'] = df['Month'].str[:4]
df['Month_Num'] = df['Month'].str[5:7].astype(int)

# 5. Encode 'Crime type' for modeling
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Crime_type_encoded'] = le.fit_transform(df['Crime type'])

# 6. Remove whitespace from strings
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 7. Drop duplicate rows
df.drop_duplicates(inplace=True)

# 8. Save cleaned data back to Desktop
df.to_excel("/Users/moeedazam/Desktop/cleaned_yorkshire_data.xlsx", index=False)

print("✅ Cleaning complete. File saved to Desktop as 'cleaned_yorkshire_data.xlsx'")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel("/Users/moeedazam/Desktop/cleaned_yorkshire_data.xlsx")

# Top 10 crime types
top_crimes = df['Crime type'].value_counts().head(10)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_crimes.values, y=top_crimes.index, palette="viridis")
plt.title("Top 10 Most Common Crime Types")
plt.xlabel("Number of Crimes")
plt.ylabel("Crime Type")
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset from Desktop
df = pd.read_excel("/Users/moeedazam/Desktop/cleaned_yorkshire_data.xlsx")

# Convert 'Month' column to datetime if not already
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')

# Count crimes by month
monthly_counts = df['Month'].dt.to_period('M').value_counts().sort_index()

# Plot
plt.figure(figsize=(12, 6))
monthly_counts.plot(kind='bar', color='skyblue')
plt.title("Total Crimes per Month")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_excel("/Users/moeedazam/Desktop/cleaned_yorkshire_data.xlsx")

# Top 10 crime locations
top_locations = df['Location'].value_counts().head(10)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_locations.values, y=top_locations.index, palette="magma")
plt.title("Top 10 Crime Locations")
plt.xlabel("Number of Crimes")
plt.ylabel("Location")
plt.tight_layout()
plt.show()

# Remove ambiguous "On or near" from location values
filtered_df = df[~df['Location'].str.strip().eq("On or near")]

# Top 10 valid locations
top_locations = filtered_df['Location'].value_counts().head(10)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_locations.values, y=top_locations.index, palette="magma")
plt.title("Top 10 Specific Crime Locations (excluding 'On or near')")
plt.xlabel("Number of Crimes")
plt.ylabel("Location")
plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the cleaned data f
df = pd.read_excel("/Users/moeedazam/Desktop/cleaned_yorkshire_data.xlsx")

# Keeping only valid latitude and longitude values
df_geo = df[['Latitude', 'Longitude']].dropna()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df_geo['Cluster'] = kmeans.fit_predict(df_geo)

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df_geo['Longitude'], df_geo['Latitude'], c=df_geo['Cluster'], cmap='tab10', alpha=0.6)
plt.title("Crime Hotspots in North Yorkshire (KMeans Clustering)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd

# Load the dataset
df = pd.read_excel("/Users/moeedazam/Desktop/cleaned_yorkshire_data.xlsx")

# Function to remove outliers using IQR
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Drop rows with missing Latitude or Longitude
df = df.dropna(subset=['Latitude', 'Longitude'])

# Remove outliers
df = remove_outliers_iqr(df, 'Latitude')
df = remove_outliers_iqr(df, 'Longitude')

# Save to a new Excel file
df.to_excel("/Users/moeedazam/Desktop/geo_outliers_removed.xlsx", index=False)

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_excel("/Users/moeedazam/Desktop/geo_outliers_removed.xlsx")

# Extract valid coordinates
df_geo = df[['Latitude', 'Longitude']].dropna()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df_geo['Cluster'] = kmeans.fit_predict(df_geo)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df_geo['Longitude'], df_geo['Latitude'], c=df_geo['Cluster'], cmap='tab10', alpha=0.6)
plt.title("Crime Hotspots in North Yorkshire (Post-Outlier Removal)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel("/Users/moeedazam/Desktop/geo_outliers_removed.xlsx")

# Encode the 'Crime type' column (if not already encoded)
le = LabelEncoder()
df['Crime_type_encoded'] = le.fit_transform(df['Crime type'])

# Features and target
X = df[['Latitude', 'Longitude', 'Month_Num']]
y = df['Crime_type_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# regression
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load your cleaned data
df = pd.read_excel("/Users/moeedazam/Desktop/geo_outliers_removed.xlsx")

# Aggregate to get crime count per LSOA
crime_counts = df.groupby(['LSOA code', 'Latitude', 'Longitude']).size().reset_index(name='Crime_Count')

# Prepare features and target
X = crime_counts[['Latitude', 'Longitude']]  # You can add more features here
y = crime_counts['Crime_Count']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("🔍 Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("📈 R² Score:", r2_score(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_excel("/Users/moeedazam/Desktop/geo_outliers_removed.xlsx")

# Encode categorical target
le = LabelEncoder()
df['Crime_type_encoded'] = le.fit_transform(df['Crime type'])

# Prepare features
X = df[['Month_Num', 'Latitude', 'Longitude']]
y = df['Crime_type_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=False, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load cleaned data
df = pd.read_excel("/Users/moeedazam/Desktop/cleaned_yorkshire_data.xlsx")

# Aggregate crime counts per month
monthly_counts = df.groupby('Month_Num').size().reset_index(name='Crime_Count')

# Features and target
X = monthly_counts[['Month_Num']]
y = monthly_counts['Crime_Count']

# Polynomial transformation
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("📉 Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("📊 R² Score:", r2_score(y_test, y_pred))

import matplotlib.pyplot as plt

# Scatter plot of predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("📊 Actual vs Predicted Crime Counts (Polynomial Regression)")
plt.xlabel("Actual Crime Count")
plt.ylabel("Predicted Crime Count")
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load and parse data
df = pd.read_excel("/Users/moeedazam/Desktop/cleaned_yorkshire_data.xlsx")

# Group crimes by Month
monthly_crimes = df['Month'].value_counts().sort_index()
monthly_crimes.index = pd.to_datetime(monthly_crimes.index)

# Plot time series
plt.figure(figsize=(10, 5))
monthly_crimes.plot()
plt.title("🕒 Monthly Crime Volume")
plt.ylabel("Number of Crimes")
plt.grid(True)
plt.tight_layout()
plt.show()

# Fit ARIMA model
model = ARIMA(monthly_crimes, order=(1, 1, 1))  # p,d,q
model_fit = model.fit()

# Forecast next 6 months
forecast = model_fit.forecast(steps=6)
print("📈 Forecasted Crimes for Next 6 Months:")
print(forecast)

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Load data (update the path if needed)
df = pd.read_excel("/Users/moeedazam/Desktop/cleaned_yorkshire_data.xlsx")

# Parse datetime and group by month
df['Month'] = pd.to_datetime(df['Month'])
monthly_crimes = df.groupby(df['Month'].dt.to_period('M')).size()
monthly_crimes.index = monthly_crimes.index.to_timestamp()

# Optional: Decomposition (for analysis)
decompose_result = seasonal_decompose(monthly_crimes, model='additive', period=12)
decompose_result.plot()
plt.tight_layout()
plt.show()

# Fit SARIMA model (adjust order as needed)
model = SARIMAX(monthly_crimes, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Forecast next 6 months
forecast = results.get_forecast(steps=6)
predicted_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(12, 5))
plt.plot(monthly_crimes, label='Observed')
plt.plot(predicted_mean, label='Forecast', color='orange')
plt.fill_between(predicted_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.2)
plt.title("📉 Crime Volume Forecast for Next 6 Months")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



