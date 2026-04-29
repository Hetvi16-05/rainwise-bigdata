import pandas as pd
import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['rainwise_db']
col = db['city_summaries']

df = pd.read_csv('data/processed/bi_dashboard_ready_PERFECT.csv')
city_stats = df.groupby('city').agg({'elevation_m': 'mean', 'Flood_Risk_Score': 'mean'}).reset_index()

for _, row in city_stats.iterrows():
    col.update_many(
        {'city': row['city']},
        {'$set': {'elevation_m': row['elevation_m'], 'Flood_Risk_Score': row['Flood_Risk_Score']}}
    )
print("MongoDB Updated!")
