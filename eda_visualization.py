import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the cleaned dataset for visualization
file_path = "cleaned_youtube.csv"
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())

# Display the first few rows of the dataset
print("\nFirst 5 Rows:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Generate basic statistics
print("\nDescriptive Statistics:")
print(data.describe())

# Visualization 1: Distribution of a numerical column (e.g., views)
if 'views' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data['views'], bins=30, kde=True)
    plt.title('Distribution of Views')
    plt.xlabel('Views')
    plt.ylabel('Frequency')
    plt.show()

# Visualization 2: Count plot of a categorical column (e.g., category)
if 'category' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(y='category', data=data, order=data['category'].value_counts().index)
    plt.title('Count of Videos by Category')
    plt.xlabel('Count')
    plt.ylabel('Category')
    plt.show()

# Visualization 3: Scatter plot between two numerical columns (e.g., views and likes)
if 'views' in data.columns and 'likes' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='views', y='likes', data=data)
    plt.title('Views vs Likes')
    plt.xlabel('Views')
    plt.ylabel('Likes')
    plt.show()

# Visualization 4: Correlation heatmap
plt.figure(figsize=(12, 8))
numeric_data = data.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()