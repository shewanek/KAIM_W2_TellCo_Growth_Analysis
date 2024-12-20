import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class UserOverview:
    def __init__(self, df):
        """Initialize with telecom data DataFrame"""
        self.df = df
        self.user_metrics = None
        plt.style.use('seaborn-v0_8')

    def select_essential_columns(self):
        """
        Select essential columns based on data completeness and analysis requirements.
        
        Selection criteria:
        1. High Completeness (>99% non-null):
           - Core identifiers, timing data, application traffic, device info
        2. Moderate Completeness (80-99%):
           - Network performance metrics
        3. Low but Important:
           - TCP retransmission metrics for QoS analysis
        
        Returns:
            pandas.DataFrame: DataFrame with selected essential columns
        """
        essential_columns = [
            # Core User/Session Identification (>99% complete)
            'Bearer Id',          # Session tracking (99.3%)
            'Start', 'End',       # Session timing (100%)
            'Dur. (s)',          # Duration (100%) 
            'MSISDN/Number',     # User ID (99.3%)
            'IMSI', 'IMEI',      # Device ID (99.6%)
            
            # Network Performance Metrics
            'Avg Bearer TP DL (kbps)',  # Throughput (100%)
            'Avg Bearer TP UL (kbps)',
            'Avg RTT DL (ms)',          # Latency (~81%)
            'Avg RTT UL (ms)',
            'TCP DL Retrans. Vol (Bytes)', # QoS analysis
            'TCP UL Retrans. Vol (Bytes)',
            
            # Application Usage (100% complete)
            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)', 
            'Email DL (Bytes)', 'Email UL (Bytes)',
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)',
            'Total DL (Bytes)', 'Total UL (Bytes)',
            
            # Device Information (99.6% complete)
            'Handset Manufacturer',
            'Handset Type',
            
            # Network Quality Indicators (~99.5% complete)
            'DL TP < 50 Kbps (%)',
            '50 Kbps < DL TP < 250 Kbps (%)',
            'UL TP < 10 Kbps (%)', 
            '10 Kbps < UL TP < 50 Kbps (%)'
        ]
        
        # Select only essential columns from DataFrame
        self.df = self.df[essential_columns].copy()
        
        return self.df


    def wrangle(self):
        """Clean and prepare data by handling missing values appropriately"""
        # Fill missing values with the mean for numeric columns
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        # Replace missing categorical values with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_value = self.df[col].mode()
            if not mode_value.empty:
                self.df[col].fillna(mode_value[0], inplace=True)

        return self.df

    def get_top_handsets(self, n=10):
        """Get top n handsets used by customers with market share percentage and visualize"""
        handset_counts = self.df['Handset Type'].value_counts()
        top_handsets = handset_counts.head(n)
        market_share = (top_handsets / len(self.df)) * 100
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        top_handsets.plot(kind='bar')
        plt.title('Top 10 Handset Types')
        plt.xlabel('Handset Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Print insights
        print("\nHandset Type Analysis:")
        print(f"- Most popular handset is {top_handsets.index[0]} with {top_handsets.iloc[0]} users")
        print(f"- Top 3 handsets account for {market_share[:3].sum():.1f}% of all users")
        
        return pd.DataFrame({
            'Count': top_handsets,
            'Market Share %': market_share.round(2)
        })

    def get_top_manufacturers(self, n=3):
        """Get top n handset manufacturers with market share percentage and visualize"""
        manufacturer_counts = self.df['Handset Manufacturer'].value_counts()
        top_manufacturers = manufacturer_counts.head(n)
        market_share = (top_manufacturers / len(self.df)) * 100
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(market_share, labels=top_manufacturers.index, autopct='%1.1f%%')
        plt.title('Market Share of Top Handset Manufacturers')
        plt.axis('equal')
        
        # Print insights
        print("\nManufacturer Analysis:")
        print(f"- Market leader is {top_manufacturers.index[0]} with {market_share.iloc[0]:.1f}% market share")
        print(f"- Top 3 manufacturers control {market_share.sum():.1f}% of the market")
        
        return pd.DataFrame({
            'Count': top_manufacturers,
            'Market Share %': market_share.round(2)
        })

    def get_top_handsets_per_manufacturer(self, n_manufacturers=3, n_handsets=5):
        """Get top n handsets for each of the top k manufacturers with market share and visualize"""
        # Get top manufacturers without plotting
        manufacturer_counts = self.df['Handset Manufacturer'].value_counts()
        top_manufacturers = manufacturer_counts.head(n_manufacturers).index
        results = {}
        
        for manufacturer in top_manufacturers:
            mask = self.df['Handset Manufacturer'] == manufacturer
            manufacturer_df = self.df[mask]
            handset_counts = manufacturer_df['Handset Type'].value_counts().head(n_handsets)
            market_share = (handset_counts / len(manufacturer_df)) * 100
            
            # Create individual plot for each manufacturer
            plt.figure(figsize=(12, 6))
            handset_counts.plot(kind='bar')
            plt.title(f'Top {n_handsets} Handsets for {manufacturer}')
            plt.xlabel('Handset Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            results[manufacturer] = pd.DataFrame({
                'Count': handset_counts,
                'Market Share %': market_share.round(2)
            })
            
            # Print insights
            print(f"\n{manufacturer} Analysis:")
            print(f"- Most popular model: {handset_counts.index[0]}")
            print(f"- Top model market share: {market_share.iloc[0]:.1f}%")
        

