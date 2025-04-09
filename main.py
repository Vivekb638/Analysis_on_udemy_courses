import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("udemy_courses.csv")

# Cleaning
df.drop_duplicates(inplace=True)
df['price'] = df['price'].replace('Free', 0).astype(int)
df['published_timestamp'] = pd.to_datetime(df['published_timestamp'], errors='coerce')

#EDA
print("Summary of dataset:\n")
print(df.info())
print("Summary Statistics:\n")
print(df.describe())

print(" Dataset Shape:", df.shape)
print("\n Column Names:\n", df.columns.tolist())
print("\n Data Types:\n", df.dtypes)
print("\n Missing Values:\n", df.isnull().sum())
print("\n Duplicate Rows:", df.duplicated().sum())

#correlaation
correlation_matrix = df.corr(numeric_only=True)
sns.set(style="white", font_scale=1.1)
plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation_matrix,
    annot=True,
    linewidths=0.5,
)
plt.title("Correlation Heatmap of Numeric Features", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

#outliers
numeric_cols = df.select_dtypes(include=[np.number]).columns
z_scores = df[numeric_cols].apply(zscore)
z_threshold = 3
outliers_by_column = {}

for col in numeric_cols:
    outliers_count = (np.abs(z_scores[col]) > z_threshold).sum()
    if outliers_count > 0:
        outliers_by_column[col] = outliers_count

print("Outliers detected using Z-score method:")
for col, count in outliers_by_column.items():
    print(f"{col}: {count} outliers")
