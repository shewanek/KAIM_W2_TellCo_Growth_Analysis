import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import mysql.connector

class SatisfactionAnalysis:
    def __init__(self, df):
        self.df = df
        self.engagement_data = df[['Total DL (Bytes)', 'Total UL (Bytes)', 'Dur. (s)']]
        self.experience_data = df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
                                   'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                                   'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']]
        self.msisdn_column = df['MSISDN/Number']

        self.engagement_kmeans = KMeans(n_clusters=3, random_state=42).fit(self.engagement_data)
        self.experience_kmeans = KMeans(n_clusters=3, random_state=42).fit(self.experience_data)

        self.scores_df = self.assign_scores()
        self.scores_df, self.top_10_satisfied = self.calculate_satisfaction_scores()
        self.model, self.mse, self.r2 = self.build_regression_model()
        self.clustered_df, self.avg_satisfaction, self.avg_experience = self.cluster_satisfaction()

    def assign_scores(self):
        scaler = StandardScaler()
        engagement_normalized = scaler.fit_transform(self.engagement_data)
        experience_normalized = scaler.fit_transform(self.experience_data)

        least_engaged_centroid = self.engagement_kmeans.cluster_centers_[np.argmin(
            np.sum(self.engagement_kmeans.cluster_centers_, axis=1))]
        worst_experience_centroid = self.experience_kmeans.cluster_centers_[np.argmin(
            np.sum(self.experience_kmeans.cluster_centers_, axis=1))]

        engagement_scores = np.linalg.norm(engagement_normalized - least_engaged_centroid, axis=1)
        experience_scores = np.linalg.norm(experience_normalized - worst_experience_centroid, axis=1)

        return pd.DataFrame({
            'MSISDN': self.msisdn_column,
            'Engagement Score': engagement_scores,
            'Experience Score': experience_scores
        })

    def calculate_satisfaction_scores(self):
        self.scores_df['Satisfaction Score'] = (self.scores_df['Engagement Score'] +
                                                self.scores_df['Experience Score']) / 2
        top_10_satisfied = self.scores_df.nlargest(10, 'Satisfaction Score')
        return self.scores_df, top_10_satisfied

    def build_regression_model(self):
        X = self.scores_df[['Engagement Score', 'Experience Score']]
        y = self.scores_df['Satisfaction Score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return model, mse, r2

    def cluster_satisfaction(self):
        X = self.scores_df[['Engagement Score', 'Experience Score']]
        kmeans = KMeans(n_clusters=2, random_state=42).fit(X)

        self.scores_df['Cluster'] = kmeans.labels_
        avg_satisfaction = self.scores_df.groupby('Cluster')['Satisfaction Score'].mean().tolist()
        avg_experience = self.scores_df.groupby('Cluster')['Experience Score'].mean().tolist()

        return self.scores_df, avg_satisfaction, avg_experience

    @staticmethod
    def plot_clusters(scores_df):
        plt.figure(figsize=(10, 6))
        plt.scatter(scores_df['Engagement Score'], scores_df['Experience Score'], c=scores_df['Cluster'], cmap='viridis')
        plt.xlabel('Engagement Score')
        plt.ylabel('Experience Score')
        plt.title('User Clusters based on Engagement and Experience')
        plt.colorbar(label='Cluster')
        plt.show()

    @staticmethod
    def export_to_mysql(scores_df, db_name, table_name):
        connection = mysql.connector.connect(host="localhost", user="root", password="SH36essti")
        cursor = connection.cursor()

        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        cursor.execute(f"USE {db_name}")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                MSISDN VARCHAR(255),
                Engagement_Score FLOAT,
                Experience_Score FLOAT,
                Satisfaction_Score FLOAT
            )
        """)

        for _, row in scores_df.iterrows():
            cursor.execute(f"""
                INSERT INTO {table_name} (MSISDN, Engagement_Score, Experience_Score, Satisfaction_Score)
                VALUES (%s, %s, %s, %s)
            """, (row['MSISDN'], row['Engagement Score'], row['Experience Score'], row['Satisfaction Score']))

        connection.commit()
        connection.close()

    @staticmethod
    def read_from_mysql(db_name, table_name):
        try:
            connection = mysql.connector.connect(host="localhost", user="root", password="oloteme", database=db_name)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, connection)
            connection.close()
            return df
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return None
