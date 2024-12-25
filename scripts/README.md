# KAIM_W2_TellCo_Growth_Analysis

## Overview

This project focuses on analyzing telecom data to gain insights into user behavior, handset usage, and network performance. The analysis is performed using Python, leveraging libraries such as pandas, matplotlib, seaborn, and scikit-learn.

## Installation

To run the code, you need to have Python installed along with the following libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Usage

The main class in this project is `UserOverview`, which provides various methods to analyze the telecom data. Below is a brief description of each method:

### Initialization

```python
import pandas as pd
from user_overview import UserOverview

# Load your data into a DataFrame
df = pd.read_csv('path_to_your_data.csv')

# Initialize the UserOverview class
user_overview = UserOverview(df)
```

### Select Essential Columns

```python
# Select essential columns based on data completeness and analysis requirements
essential_df = user_overview.select_essential_columns()
```

### Data Wrangling

```python
# Clean and prepare data by handling missing values appropriately
cleaned_df = user_overview.wrangle()
```

### Handset Analysis

```python
# Get top 10 handsets used by customers
top_handsets = user_overview.get_top_handsets(n=10)

# Get top 3 handset manufacturers
top_manufacturers = user_overview.get_top_manufacturers(n=3)

# Get top 5 handsets for each of the top 3 manufacturers
user_overview.get_top_handsets_per_manufacturer(n_manufacturers=3, n_handsets=5)
```

### User Behavior Analysis

```python
# Aggregate user behavior metrics per user
user_metrics = user_overview.aggregate_user_behavior()

# Segment users into deciles based on total duration
decile_stats = user_overview.get_user_deciles()
```

### Statistical Metrics

```python
# Compute and visualize basic statistical metrics for numeric columns
stats = user_overview.compute_basic_metrics()
```

### Correlation Analysis

```python
# Compute and visualize correlation matrix for application data
corr_matrix = user_overview.compute_correlation_matrix()
```

### Principal Component Analysis (PCA)

```python
# Perform PCA on application data and visualize results
pca_results = user_overview.perform_pca(n_components=2)
```

### Save Data

```python
# Save the user metrics DataFrame to a CSV file
user_overview.save_data(filename='clean_processed_data.csv')
```

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.


## Acknowledgements

Special thanks to 10 Academy for providing the data and guidance for this project.
# Task 2
# **User Engagement Analysis Toolkit**

This toolkit provides Python-based solutions for analyzing user engagement and application traffic data. It helps businesses gain insights into user behavior, application usage, and clustering patterns.

## **Features**
- Aggregate session metrics (frequency, duration, and traffic) for each customer.
- Identify the top 10 customers per metric (session frequency, duration, traffic).
- Perform K-Means clustering to segment users.
- Analyze and visualize traffic data across various applications.
- Generate insightful plots, including the elbow method and top-used applications.

## **Requirements**
Install the required Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## **Usage**

### **1. Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/shewanek/KAIM_W2_TellCo_Growth_Analysis.git
   cd user-engagement-analysis
   ```
2. Ensure your dataset is in `.csv` or `.xlsx` format and contains necessary columns (e.g., `MSISDN/Number`, `Dur. (s)`, `Total DL (Bytes)`, etc.).

### **2. Run Analysis**
```python
from user_engagement_analysis import UserEngagementAnalysis


# Initialize analysis object
analysis = UserEngagementAnalysis(data)

# Perform analysis
aggregated_data = analysis.aggregate_metrics_per_customer()
top_10 = analysis.top_10_customers_per_metric(aggregated_data)
normalized_data = analysis.normalize_metrics(aggregated_data)
clusters = analysis.k_means_clustering(normalized_data)
app_traffic = analysis.aggregate_traffic_per_application()
analysis.plot_top_3_applications(app_traffic)
```

### **Outputs**
- Aggregated session metrics (frequency, duration, traffic).
- Clustering-based user segmentation.
- Traffic reports for applications like YouTube, Social Media, and Google.

## **Functions Overview**
| Function                              | Description                                                 |
|---------------------------------------|-------------------------------------------------------------|
| `aggregate_metrics_per_customer()`    | Aggregate session metrics per customer.                     |
| `top_10_customers_per_metric()`       | Identify the top 10 customers for each key metric.          |
| `normalize_metrics()`                 | Normalize metrics for clustering.                           |
| `k_means_clustering()`                | Perform user segmentation with K-means clustering.          |
| `aggregate_traffic_per_application()` | Aggregate and analyze traffic data for various applications.|
| `plot_top_3_applications()`           | Visualize traffic for the top 3 most-used applications.     |
# Satisfaction Analysis

A Python script for analyzing customer satisfaction data and generating insights.

## Overview

This script performs sentiment analysis on customer feedback data to measure satisfaction levels and identify trends. It processes text data to extract sentiment scores and provides visualizations of the results.

## Features

- Sentiment analysis of customer feedback
- Data preprocessing and cleaning
- Statistical analysis of satisfaction scores
- Visualization of trends and patterns
- Export of results to various formats

