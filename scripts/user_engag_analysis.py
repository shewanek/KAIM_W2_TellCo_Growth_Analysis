import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

class UserEngagementAnalysis:
    def __init__(self, data):
        self.data = data

    def aggregate_metrics_per_customer(self):
        self.data['session_duration'] = self.data['Dur. (s)']
        self.data['total_traffic'] = self.data['Total DL (Bytes)'] + self.data['Total UL (Bytes)']
        self.data['session_frequency'] = self.data.groupby('MSISDN/Number')['Bearer Id'].transform('count')
        
        aggregated_data = self.data.groupby('MSISDN/Number').agg({
            'session_frequency': 'sum',
            'session_duration': 'sum',
            'total_traffic': 'sum'
        }).reset_index()
        return aggregated_data

    def top_10_customers_per_metric(self, aggregated_data):
        top_10_frequency = "top_10_frequency", aggregated_data.nlargest(10, 'session_frequency')
        top_10_duration = "top_10_duration", aggregated_data.nlargest(10, 'session_duration')
        top_10_traffic = "top_10_traffic", aggregated_data.nlargest(10, 'total_traffic')
        return top_10_frequency, top_10_duration, top_10_traffic

    