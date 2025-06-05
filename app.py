#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 00:35:52 2025

@author: moeedazam
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.linear_model import LinearRegression

# Page setup
st.set_page_config(page_title="North Yorkshire Crime Dashboard", layout="wide")
st.title("North Yorkshire Crime Analytics Dashboard")

# Load data
df = pd.read_excel("geo_outliers_removed.xlsx")

# Sidebar filters
st.sidebar.header("🔍 Filters")
years = df['Year'].unique()
selected_year = st.sidebar.selectbox("Select Year", sorted(years, reverse=True))
crime_types = df[df['Year'] == selected_year]['Crime type'].unique()
selected_crimes = st.sidebar.multiselect("Select Crime Type(s)", sorted(crime_types), default=crime_types)

# Filter data
df_filtered = df[(df['Year'] == selected_year) & (df['Crime type'].isin(selected_crimes))]

# CSV Download
st.sidebar.markdown("### 📁 Download Filtered Data")
def convert_df(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()

st.sidebar.download_button(
    label="Download CSV",
    data=convert_df(df_filtered),
    file_name='filtered_crime_data.csv',
    mime='text/csv'
)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔹 Overview", "📈 Time Series Forecast", "📉 Regression", "🗺️ Crime Map", "📌 Clustering"
])

with tab1:
    # Top Crime Types
    st.subheader(f"Top Crime Types in {selected_year}")
    top_crimes = df_filtered['Crime type'].value_counts().head(10)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=top_crimes.values, y=top_crimes.index, palette="viridis", ax=ax1)
    ax1.set_xlabel("Number of Crimes")
    ax1.set_ylabel("Crime Type")
    st.pyplot(fig1)

    # Top Locations
    st.subheader("Top 10 Specific Crime Locations")
    filtered_location = df_filtered[~df_filtered['Location'].str.strip().eq("On or near")]
    top_locations = filtered_location['Location'].value_counts().head(10)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=top_locations.values, y=top_locations.index, palette="magma", ax=ax2)
    ax2.set_xlabel("Number of Crimes")
    ax2.set_ylabel("Location")
    st.pyplot(fig2)

    # Monthly Trend
    st.subheader("📅 Monthly Crime Trend")
    monthly = df_filtered.groupby("Month")['Crime ID'].count().reset_index()
    fig3, ax3 = plt.subplots()
    sns.lineplot(data=monthly, x='Month', y='Crime ID', marker='o', ax=ax3)
    ax3.set_ylabel("Number of Crimes")
    ax3.set_xticklabels(monthly['Month'], rotation=45)
    st.pyplot(fig3)

with tab2:
    # Time Series Forecast
    st.subheader("🔮 Forecasted Monthly Crime Volume")
    df_ts = df[df['Year'] == selected_year]
    ts_data = df_ts.groupby("Month")['Crime ID'].count().reset_index()
    ts_data = ts_data.sort_values('Month')
    model = SimpleExpSmoothing(ts_data['Crime ID'])
    fit = model.fit(smoothing_level=0.6, optimized=False)
    ts_data['Forecast'] = fit.fittedvalues
    fig4 = px.line(ts_data, x='Month', y=['Crime ID', 'Forecast'],
                   labels={'value': 'Crime Count', 'Month': 'Month'},
                   title='Crime Volume Forecast')
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    # Regression
    st.subheader("📉 Linear Regression: Crime Volume Trend Over Time")
    monthly_data = df.groupby(['Year', 'Month']).size().reset_index(name='Total_Crimes')
    monthly_data['Month_Ordinal'] = range(len(monthly_data))
    model = LinearRegression()
    model.fit(monthly_data[['Month_Ordinal']], monthly_data[['Total_Crimes']])
    monthly_data['Prediction'] = model.predict(monthly_data[['Month_Ordinal']])
    fig5, ax5 = plt.subplots()
    sns.scatterplot(data=monthly_data, x='Month_Ordinal', y='Total_Crimes', label='Actual', color='blue')
    sns.lineplot(data=monthly_data, x='Month_Ordinal', y='Prediction', label='Regression Line', color='red')
    ax5.set_title("📉 Crime Volume Regression with Month Ordinal Encoding")
    ax5.set_xlabel("Month Ordinal")
    ax5.set_ylabel("Total Crimes")
    st.pyplot(fig5)

with tab4:
    # Crime Map
    st.subheader("🗺️ Crime Distribution Map")
    if 'Latitude' in df_filtered.columns and 'Longitude' in df_filtered.columns:
        map_data = df_filtered[['Latitude', 'Longitude']].dropna()
        map_data = map_data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
        st.map(map_data)
    else:
        st.error("No Latitude/Longitude data available in filtered results.")






