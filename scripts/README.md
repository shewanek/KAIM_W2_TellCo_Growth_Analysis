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
