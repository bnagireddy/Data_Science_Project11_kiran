import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Load the dataset
df = pd.read_csv("youtube.csv", encoding='latin1')

# STEP 2: Preview the data
print("Original Data:")
print(df.head())
print("\nColumns:", df.columns)

# STEP 3: Drop duplicates and unnecessary columns
df.drop_duplicates(inplace=True)

# STEP 4: Combine 'publish_date' and 'time_frame' into a single datetime column
if 'publish_date' in df.columns and 'time_frame' in df.columns:
    df['publish_datetime'] = pd.to_datetime(df['publish_date'] + ' ' + df['time_frame'], errors='coerce')
else:
    print("Missing 'publish_date' or 'time_frame' columns.")

# STEP 5: Extract new datetime features
df['publish_hour'] = df['publish_datetime'].dt.hour
df['publish_day'] = df['publish_datetime'].dt.day
df['publish_month'] = df['publish_datetime'].dt.month
df['publish_day_of_week'] = df['publish_datetime'].dt.day_name()

# STEP 6: Clean string/text fields (if needed)
df['tags'] = df['tags'].str.replace('"', '').str.replace('[', '').str.replace(']', '')
df['title'] = df['title'].str.strip()

# STEP 7: Fill missing numerical values (if any)
num_cols = ['views', 'likes', 'dislikes', 'comment_count']
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0)

# STEP 8: Save cleaned data
df.to_csv("cleaned_youtube.csv", index=False)
print("\n✅ Data cleaning complete. Saved as 'cleaned_youtube.csv'")
