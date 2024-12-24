# **User Engagement Analysis Toolkit**

This toolkit provides a comprehensive solution for analyzing user engagement, traffic metrics, and application usage patterns. It uses advanced data aggregation, normalization, and clustering techniques to deliver actionable insights for businesses and network optimization.

---

## **Key Features**
- **User Engagement Metrics**:  
  Analyze session frequency, session duration, and total traffic per customer.
  
- **Top User Analysis**:  
  Identify top 10 customers based on session frequency, session duration, and total traffic.

- **Traffic Aggregation by Application**:  
  Analyze data usage for applications like YouTube, Gaming, Email, and Social Media.

- **K-Means Clustering**:  
  Segment users into behavioral groups with support for the elbow method to determine optimal clusters.

- **Visualization**:  
  Generate plots for top-used applications, user engagement metrics, and clustering insights.

---

## **Installation**

### **Requirements**
- Python 3.7+
- Libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```


### **1. Analyze User Engagement**
```python
import pandas as pd
from scripts.user_engag_analysis import UserEngagementAnalysis

# Load dataset
df = pd.read_csv('../data/clean_proccessed_data.csv')

# Initialize User Engagement Analysis
User_Engag = UserEngagementAnalysis(df)

# Aggregate metrics
aggregated_data = User_Engag.aggregate_metrics_per_customer()

# Get top 10 users
top_10_frequency, top_10_duration, top_10_traffic = User_Engag.top_10_customers_per_metric(aggregated_data)
print(top_10_frequency)
```

### **3. Application Traffic Analysis**
```python
# Aggregate traffic by application
app_traffic = User_Engag.aggregate_traffic_per_application()

# Top 10 users per application
top_10_users = User_Engag.top_10_users_per_application(app_traffic)
print(top_10_users)
```

### **4. Clustering and Visualization**
```python
# Normalize metrics
normalized_data = User_Engag.normalize_metrics(aggregated_data)

# Perform K-means clustering
clusters = User_Engag.k_means_clustering(normalized_data, k=3)

# Compute cluster metrics
cluster_metrics = User_Engag.compute_cluster_metrics(aggregated_data, clusters)
print(cluster_metrics)

# Plot elbow method
User_Engag.plot_elbow_method(normalized_data)

# Visualize top 3 applications
User_Engag.plot_top_3_applications(app_traffic)
```

---

## **Key Insights**

### **User Engagement Metrics**
1. **High Session Frequency**:  
   - The most engaged user has 1,136,356 sessions.
   - Indicates frequent network usage for targeted retention strategies.

2. **Top Data Users**:  
   - The top customer generated 531 TB of total traffic.
   - Suggests optimization for heavy users to improve network efficiency.

---

### **Application Traffic Patterns**
1. **Gaming Dominance**:  
   - Gaming apps consume the most data (~63.3 PB).  
   - Recommend network optimization and gaming-specific QoS policies.

2. **Other Applications**:  
   - "Other" category consumes ~63.2 PB.  
   - Requires detailed classification for better insights.

3. **YouTube Usage**:  
   - Accounts for only ~1.4% of top traffic.  
   - Indicates opportunities for video streaming services.

---

### **Clustering Results**
- Behavioral segmentation reveals distinct user groups:
  - **High-frequency, low-traffic users**.
  - **Moderate-frequency, high-traffic users**.
  - **Low-frequency, high-duration users**.
- Supports personalized marketing and resource allocation.

