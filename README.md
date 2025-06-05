 North Yorkshire Crime Analytics Dashboard

This Streamlit-powered dashboard provides interactive visual insights into crime data across North Yorkshire. It allows users to filter by year and crime type, explore trends, and visualize crime distribution using multiple analytics views.

---

## 📁 Dataset
The dashboard uses a pre-cleaned dataset:  
**`geo_outliers_removed.xlsx`** — Includes crime records with geolocation and time attributes, filtered to remove extreme outliers.

---

## 📌 Features

### 🔍 Filters
- Select crime **year** and **crime types** from sidebar
- Download filtered dataset as CSV

### 🔹 Overview Tab
- Top 10 crime types
- Top 10 specific locations
- Monthly crime trend plot

### 📈 Time Series Forecast
- Forecast future monthly crime volume using exponential smoothing

### 📉 Regression Analysis
- Linear regression to analyze long-term crime volume trend

### 🗺️ Crime Map
- Map of crime incidents by location using latitude/longitude data

---

## 🚀 Deployment
Deployed via **Streamlit Cloud**  
[🔗 Click to view live app](https://north-yorkshire-crime-dashboard-uhyw2ykj8etdcmtvrdahsy.streamlit.app/)

---

## 🧪 Requirements

Install packages with:

```bash
pip install -r requirements.txt

Moeed Azam
MSc Business Analytics & Management — University of East Anglia
