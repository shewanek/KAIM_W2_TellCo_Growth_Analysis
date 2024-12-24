import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


class TelecomAnalyzer:
    def __init__(self, df):
        self.df = df
        self.agg_df = None
        self.clustered_df = None

    def get_value_stats(self, series, n=10):
        return pd.DataFrame({
            'Top': series.nlargest(n).values,
            'Bottom': series.nsmallest(n).values,
            'Most Frequent': series.value_counts().nlargest(n).index
        })
    
    def analyze_distributions(self):
        # Throughput per handset - show only top 15 handsets
        throughput_by_handset = self.df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean().sort_values(ascending=False).head(15)

        plt.figure(figsize=(15,8))
        ax = throughput_by_handset.plot(kind='bar')
        plt.title('Average Throughput by Top 15 Handset Types', fontsize=12, pad=20)
        plt.xlabel('Handset Type', fontsize=10)
        plt.ylabel('Average Throughput (kbps)', fontsize=10)
        plt.xticks(rotation=30, ha='right')
        # Add value labels on top of bars
        for i, v in enumerate(throughput_by_handset):
            ax.text(i, v, f'{int(v):,}', ha='center', va='bottom')
        plt.tight_layout()
        
        # Display throughput table
        print("\nTop 15 Handsets by Average Throughput:")
        print(throughput_by_handset.reset_index().to_string(index=False))

        # TCP retransmission per handset - show only top 15 handsets
        tcp_by_handset = self.df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean().sort_values(ascending=False).head(15)

        plt.figure(figsize=(15,8))
        ax = tcp_by_handset.plot(kind='bar')
        plt.title('Average TCP Retransmission by Top 15 Handset Types', fontsize=12, pad=20)
        plt.xlabel('Handset Type', fontsize=10)
        plt.ylabel('Average TCP Retransmission (Bytes)', fontsize=10)
        plt.xticks(rotation=30, ha='right')
        # Add value labels on top of bars
        for i, v in enumerate(tcp_by_handset):
            ax.text(i, v, f'{int(v):,}', ha='center', va='bottom')
        plt.tight_layout()
        
        # Display TCP retransmission table
        print("\nTop 15 Handsets by Average TCP Retransmission:")
        print(tcp_by_handset.reset_index().to_string(index=False))


    def aggr_user_metrics(self):
        """
        Aggregate the following per customer (MSISDN/Number):
        - Average TCP retransmission
        - Average RTT
        - Handset type
        - Average throughput
        """
        self.agg_df = self.df.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Handset Type': lambda x: x.mode()[0]  # Most frequent Handset type
        }).rename(columns={
            'TCP DL Retrans. Vol (Bytes)': 'avg_tcp_retrans',
            'Avg RTT DL (ms)': 'avg_rtt',
            'Avg Bearer TP DL (kbps)': 'avg_throughput'
        })
        return self.agg_df
    

    def run_kmeans_clustering(self, k=3):
        """
        Perform K-Means clustering (k=3) to segment users based on their experience.
        """
        # Call aggr_user_metrics first if agg_df is None
        if self.agg_df is None:
            self.aggr_user_metrics()
            
        scaler = MinMaxScaler()
        self.agg_df[['avg_tcp_retrans', 'avg_rtt', 'avg_throughput']] = scaler.fit_transform(
            self.agg_df[['avg_tcp_retrans', 'avg_rtt', 'avg_throughput']])
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        self.agg_df['cluster'] = kmeans.fit_predict(self.agg_df[['avg_tcp_retrans', 'avg_rtt', 'avg_throughput']])
        
        self.clustered_df = self.agg_df
        
        # Visualize the clusters
        sns.pairplot(self.clustered_df, hue='cluster', diag_kind='kde', 
                     vars=['avg_tcp_retrans', 'avg_rtt', 'avg_throughput'], palette='Set2', corner=True)
        plt.suptitle('K-means Clustering Results - User Experience Clusters', y=1.02)
        plt.show()

        return self.clustered_df

    def describe_clusters(self):
        """
        Provide a description of each cluster based on user experience metrics.
        """
        cluster_description = self.clustered_df.groupby('cluster').agg({
            'avg_tcp_retrans': ['min', 'max', 'mean'],
            'avg_rtt': ['min', 'max', 'mean'],
            'avg_throughput': ['min', 'max', 'mean'],
            'Handset Type': lambda x: x.mode()[0]  # Most common handset type per cluster
        })
        print("Cluster Descriptions:\n", cluster_description)
        return cluster_description  


