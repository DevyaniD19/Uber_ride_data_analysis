# 🚗 Uber Ride Data Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Exploratory data analysis on Uber ride patterns, demand forecasting, and pricing trends using Python.

## 📌 Overview

This project analyzes a real-world Uber rides dataset to uncover insights about:
- Peak demand times and days
- Most frequent pickup/dropoff locations
- Trip duration and distance distributions
- Ride purpose breakdown (Business vs Personal)

## 🗂️ Repository Structure

```
Uber_ride_data_analysis/
├── data_analysis.ipynb     # Interactive notebook with full EDA
├── data_analysis.py        # Python script version
├── UberDataset.csv         # Dataset (Uber rides log)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 📊 Key Insights

- **Peak hours**: 7–9 AM and 5–7 PM on weekdays
- **Top purpose**: Business travel (~85% of rides)
- **Longest routes**: Airport pickups average 2× longer than city rides
- **Seasonal trends**: Higher demand in Q4 (Oct–Dec)

## ⚙️ Setup

```bash
pip install -r requirements.txt
jupyter notebook data_analysis.ipynb
```

## 👩‍💻 Author

**Devyani Deore** — [github.com/DevyaniD19](https://github.com/DevyaniD19)

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🔍 Analysis Sections

| Section | Description |
|---------|-------------|
| Data Loading | Read CSV, parse dates, inspect shape |
| Data Cleaning | Handle nulls, parse categorical fields |
| Demand Analysis | Hourly, daily, and monthly ride counts |
| Trip Distance | Distribution and outlier detection |
| Purpose Analysis | Business vs. Personal ride breakdown |
| Location Heatmap | Top start/stop categories |
