import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# STEP 1: Load the cleaned dataset
file_path = "cleaned_youtube.csv"
data = pd.read_csv(file_path)

# STEP 2: Select relevant features for training
# Define target variable and features
if 'views' in data.columns:
    target = 'views'
    features = ['likes', 'dislikes', 'comment_count']

    # Check if features exist in the dataset
    for feature in features:
        if feature not in data.columns:
            print(f"Feature '{feature}' not found in the dataset. Exiting.")
            exit()

    X = data[features]
    y = data[target]
else:
    print("Target variable 'views' not found in the dataset. Exiting.")
    exit()

# STEP 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 4: Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 5: Make predictions on the test set
y_pred = model.predict(X_test)

# STEP 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Training Complete")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# STEP 7: Save the model (optional)
import joblib
joblib.dump(model, "linear_regression_model.pkl")
print("\n✅ Model saved as 'linear_regression_model.pkl'")