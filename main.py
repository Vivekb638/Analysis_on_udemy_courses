import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("udemy_courses.csv")

# Cleaning
df.drop_duplicates(inplace=True)
df['price'] = df['price'].replace('Free', 0).astype(int)
df['published_timestamp'] = pd.to_datetime(df['published_timestamp'], errors='coerce')

# ========== 1. BASIC INFO ==========
print(" Dataset Shape:", df.shape)
print("\n Column Names:\n", df.columns.tolist())
print("\n Data Types:\n", df.dtypes)
print("\n Missing Values:\n", df.isnull().sum())
print("\n Duplicate Rows:", df.duplicated().sum())
