import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class UserEngagementAnalysis:
    def __init__(self, data):
        self.data = data


    def aggregate_metrics_per_customer(self):
        """Aggregate key metrics per customer with error handling"""
        try:
            self.data['session_duration'] = pd.to_numeric(self.data['Dur. (s)'], errors='coerce')
            self.data['total_traffic'] = pd.to_numeric(self.data['Total DL (Bytes)'], errors='coerce') + \
                                       pd.to_numeric(self.data['Total UL (Bytes)'], errors='coerce')
            self.data['session_frequency'] = self.data.groupby('MSISDN/Number')['Bearer Id'].transform('count')
            
            aggregated_data = self.data.groupby('MSISDN/Number').agg({
                'session_frequency': 'sum',
                'session_duration': 'sum',
                'total_traffic': 'sum'
            }).reset_index()
            
            # Remove any rows with null values
            aggregated_data = aggregated_data.dropna()
            return aggregated_data
        except Exception as e:
            raise RuntimeError(f"Error in aggregating metrics: {str(e)}")

    def top_10_customers_per_metric(self, aggregated_data):
        """Get top 10 customers for each metric with validation"""
        if len(aggregated_data) < 1:
            raise ValueError("Empty aggregated data")
            
        top_10_frequency = "top_10_frequency", aggregated_data.nlargest(10, 'session_frequency')
        top_10_duration = "top_10_duration", aggregated_data.nlargest(10, 'session_duration')
        top_10_traffic = "top_10_traffic", aggregated_data.nlargest(10, 'total_traffic')
        return top_10_frequency, top_10_duration, top_10_traffic

    def normalize_metrics(self, aggregated_data):
        """Normalize metrics using StandardScaler with error handling"""
        try:
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(aggregated_data[['session_frequency', 'session_duration', 'total_traffic']])
            return normalized_data
        except Exception as e:
            raise RuntimeError(f"Error in normalizing metrics: {str(e)}")

    def k_means_clustering(self, normalized_data, k=3):
        """Perform k-means clustering with input validation"""
        if k < 1:
            raise ValueError("Number of clusters must be positive")
        if len(normalized_data) < k:
            raise ValueError("More clusters than data points")
            
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(normalized_data)
        return clusters

    def compute_cluster_metrics(self, aggregated_data, clusters):
        """Compute metrics for each cluster with validation"""
        if len(aggregated_data) != len(clusters):
            raise ValueError("Mismatch between data and cluster assignments")
            
        aggregated_data['cluster'] = clusters
        cluster_metrics = aggregated_data.groupby('cluster').agg({
            'session_frequency': ['min', 'max', 'mean', 'sum'],
            'session_duration': ['min', 'max', 'mean', 'sum'],
            'total_traffic': ['min', 'max', 'mean', 'sum']
        })
        return cluster_metrics

    def plot_elbow_method(self, normalized_data, max_k=10):
        """Plot elbow method with configurable max_k"""
        if max_k < 2:
            raise ValueError("max_k must be at least 2")
            
        sse = []
        k_values = range(1, max_k + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(normalized_data)
            sse.append(kmeans.inertia_)
            
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, sse, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.title('Elbow Method For Optimal k')
        plt.grid(True)
        plt.show()

    def aggregate_traffic_per_application(self):
        """Aggregate traffic per application with error handling"""
        try:
            app_columns = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 
                          'Google DL (Bytes)', 'Google UL (Bytes)',
                          'Email DL (Bytes)', 'Email UL (Bytes)', 
                          'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                          'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
                          'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
                          'Other DL (Bytes)', 'Other UL (Bytes)']
            
            # Validate columns exist
            missing_cols = [col for col in app_columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            app_traffic = self.data.melt(id_vars=['MSISDN/Number'], 
                                       value_vars=app_columns, 
                                       var_name='application', 
                                       value_name='traffic')
            
            app_traffic['application'] = app_traffic['application'].str.replace(' DL \(Bytes\)| UL \(Bytes\)', '')
            app_traffic['traffic'] = pd.to_numeric(app_traffic['traffic'], errors='coerce')
            
            app_traffic = app_traffic.groupby(['application', 'MSISDN/Number']).agg({
                'traffic': 'sum'
            }).reset_index()
            
            return app_traffic.dropna()
        except Exception as e:
            raise RuntimeError(f"Error in aggregating traffic: {str(e)}")

    def top_10_users_per_application(self, app_traffic):
        """Get top 10 users per application with validation"""
        if len(app_traffic) < 1:
            raise ValueError("Empty application traffic data")
            
        top_10_users = app_traffic.groupby('application').apply(
            lambda x: x.nlargest(10, 'traffic')).reset_index(drop=True)
        return top_10_users

    def plot_top_3_applications(self, app_traffic):
        """Plot top 3 applications with improved visualization"""
        if len(app_traffic) < 1:
            raise ValueError("Empty application traffic data")
            
        top_3_apps = app_traffic.groupby('application')['traffic'].sum().nlargest(3)
        print("Top 3 most used applications with values:", top_3_apps.to_dict())
        
        top_3_data = app_traffic[app_traffic['application'].isin(top_3_apps.index)]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='application', y='traffic', data=top_3_data, 
                   estimator=sum, ci=None, palette='viridis')
        
        plt.title('Top 3 Most Used Applications\n' + 
                 '\n'.join([f"{app}: {value:,.0f} bytes" for app, value in top_3_apps.items()]))
        plt.xlabel('Application')
        plt.ylabel('Total Traffic (Bytes)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
