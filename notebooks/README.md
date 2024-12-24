# User Satisfaction Analysis

## Task 1 - User Overview Analysis

### Objective
Understand the dataset and identify missing values & outliers using visual and quantitative methods.

### Tasks
1. Identify the top 10 handsets used by customers.
2. Identify the top 3 handset manufacturers.
3. Identify the top 5 handsets per top 3 handset manufacturer.
4. Provide interpretation and recommendations to marketing teams.


### Initial Data Analysis Insights
- Dataset contains 150,001 records with 55 columns.
- Most columns are numeric (50 float64) with only 5 object (string) columns.
- Several columns have missing values, particularly:
  - TCP DL/UL Retrans. Vol (~60% missing)
  - HTTP DL/UL Bytes (~55% missing)
  - Volume-based time metrics (60-87% missing)
- All application traffic columns (Social Media, Google, Email etc.) have complete data.
- Core user identifiers (MSISDN, IMSI, IMEI) have ~1% missing values.
- Key metrics like duration, throughput and total bytes are mostly complete.

### Identifying Columns with High Percentage of Missing Values
```python
# Calculate the percentage of missing values in each column
missing_values = df.isnull().sum() / len(df)

# Filter the columns where more than 10% of the values are missing
high_missing_values = missing_values[missing_values > 0.1]
print(high_missing_values)
```

### Data Variables Analysis and Column Selection Strategy
#### Overview of Dataset Structure and Completeness
- 50 numeric columns (float64)
- 5 categorical columns (object)

#### Column Selection Strategy Based on Data Completeness
- **High Completeness (>99% non-null)**
  - Core identifiers (Bearer Id, IMSI, MSISDN/Number, IMEI)
  - Timing data (Start, End, Duration)
  - All application traffic metrics (Social Media, Google, Email, etc.)
  - Total traffic metrics (Total UL/DL Bytes)
  - Throughput percentages (DL/UL TP distributions)
  - Device information (Handset Manufacturer, Type)
- **Moderate Completeness (80-99% non-null)**
  - Network performance metrics (Avg RTT DL/UL)
- **Low Completeness (<80% non-null)**
  - TCP retransmission volumes (~40% non-null)
  - HTTP traffic metrics (~45% non-null)
  - Detailed volume-based time metrics (~25-40% non-null)

#### Selected Essential Columns
- **Core User/Session Identification (>99% complete)**
  - Bearer Id: Session tracking (99.3% complete)
  - Start, End, Dur. (s): Session timing (100% complete)
  - MSISDN/Number: User ID (99.3% complete)
  - IMSI, IMEI: Device ID (99.6% complete)
- **Network Performance Metrics**
  - Avg Bearer TP DL/UL (kbps): Throughput (100% complete)
  - Avg RTT DL/UL (ms): Latency (~81% complete)
  - TCP Retrans. Vol: Retained despite low completeness for QoS analysis
- **Application Usage (100% complete)**
  - Social Media, Google, Email traffic
  - Streaming (YouTube, Netflix)
  - Gaming traffic
  - Other and Total traffic
- **Device Information (99.6% complete)**
  - Handset Manufacturer
  - Handset Type
- **Network Quality Indicators (~99.5% complete)**
  - DL/UL throughput distribution percentages

### Data Quality Summary
- Core metrics maintain >99% completeness.
- Network performance metrics show varying completeness.
- Application usage data is complete.
- Device information highly reliable.

### Data Cleaning and Analysis
```python
# Initialize UserOverview class with our DataFrame
user_overv = UserOverview(df)

# Select essential columns for analysis
select_data = user_overv.select_essential_columns()
select_data.head()

# Clean and handle missing values
clean_data = user_overv.wrangle()
clean_data.head()
```

### Analyze Top Handset Types and Market Share
```python
user_overv.get_top_handsets()
```

#### Key Insights from Handset Analysis
- Huawei B528S-23A dominates with 13.55% market share.
- Apple iPhones collectively hold ~29% of the market.
- High number of 'undefined' devices (6%) suggests potential data quality issues.
- Samsung has only one model (Galaxy S8) in top 10 with 3% share.
- Top 3 handsets account for over 25% of all devices.
- Premium smartphones (iPhone X, Galaxy S8) have lower market share than mid-range models.

### Analyze Top Manufacturers and Market Share
```python
user_overv.get_top_manufacturers()
```

#### Key Insights from Manufacturer Analysis
- Apple leads with 40.1% market share.
- Samsung holds second place with 27.2% of the market.
- Huawei captures 23% market share.
- Top 3 manufacturers control over 90% of the market.
- Significant gap between Apple and others suggests premium positioning success.

### Analyze Top Handsets per Manufacturer
```python
user_overv.get_top_handsets_per_manufacturer()
```

#### Key Insights from Manufacturer-Specific Analysis
- Huawei's B528S-23A router dominates their portfolio with 57.4% share.
- Apple shows more balanced distribution with iPhone 6S leading at 15.7%.
- Samsung's Galaxy S8 leads at 11.1%, reflecting balanced mid-to-high end market presence.
- Huawei's high concentration in one model vs Apple/Samsung's distributed shares shows different market strategies.
- Legacy models (iPhone 6S, Galaxy S8) leading suggests longer replacement cycles.

### Analyze User Behavior and Application Usage
```python
user_overv.aggregate_user_behavior()
```

#### Key Insights from Application Usage Analysis
- Gaming and Other categories dominate data consumption.
- Social Media shows relatively low data usage despite likely high engagement frequency.
- Video streaming (YouTube, Netflix) shows moderate usage levels.
- Email and Google services show minimal data consumption.
- High "Other" category suggests significant uncategorized traffic that needs investigation.
- Data usage patterns indicate entertainment-focused user behavior over productivity applications.

### Analyze User Deciles
```python
user_overv.get_user_deciles()
```

#### User Decile Analysis
- Top 10% of users (D10) average 8077.2 minutes.
- Bottom 10% of users (D1) average 334.5 minutes.
- Significant disparities in user engagement, with the top decile exhibiting over 24 times the average duration of the bottom decile.

### Compute Basic Metrics
```python
user_overv.compute_basic_metrics()
```

#### Insights about the Distribution
- Significant outliers in Total DL and Total UL indicate a small number of users consume disproportionately high data.
- The duration metric shows a more consistent user engagement pattern, with fewer extreme values.

### Compute and Visualize Correlation Matrix for Application Data
```python
user_overv.compute_correlation_matrix()
```

#### Correlation Analysis
- Strong correlations among application usage indicate significant relationships.
- These correlations can inform targeted strategies for user engagement.

### Perform PCA on Application Data
```python
user_overv.perform_pca()
```

#### PCA Analysis Insights
- The first principal component (PC1) explains 97.5% of the variance.
- The total explained variance by the first two components is 98.0%.
- The loadings of PC1 show that all application usage metrics contribute almost equally.
- The second principal component (PC2) has significant loadings for Gaming and Other categories.

### Save Cleaned Data
```python
user_overv.save_data()
print("Data saved to ../data/clean_proccessed_data.csv")
```

### Recommendations to Marketing Teams
- Focus on promoting Huawei B528S-23A and Apple iPhones due to their high market share.
- Address data quality issues related to 'undefined' devices.
- Leverage the high engagement in gaming and other entertainment categories for targeted marketing campaigns.
- Consider strategies to engage the bottom decile users to increase their usage and engagement.
- Utilize the strong correlations among application usage for cross-promotional activities.
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

