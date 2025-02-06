import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the Data
try:
    df = pd.read_csv('weather.csv')
except FileNotFoundError:
    print("Error: File 'weather.csv' not found. Please check the file path.")
    exit()

# Step 2: Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Handle missing data
df = df.dropna()  # Drop rows with missing values

# Step 3: Data Visualization
if all(col in df.columns for col in ['MinTemp', 'MaxTemp', 'Rainfall']):
    sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
    plt.show()
else:
    print("Warning: One or more columns ['MinTemp', 'MaxTemp', 'Rainfall'] are missing.")

# Step 4: Feature Engineering
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # Drop rows with invalid dates
    df['Month'] = df['Date'].dt.month
else:
    print("Warning: 'Date' column not found. Feature engineering skipped.")

# Step 5: Data Analysis
if 'Month' in df.columns and 'MaxTemp' in df.columns:
    monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()

    # Step 6: Data Visualization (Part 2)
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
    plt.xlabel('Month')
    plt.ylabel('Average Max Temperature')
    plt.title('Monthly Average Max Temperature')
    plt.grid(True)
    plt.show()
else:
    print("Warning: Unable to analyze or plot monthly average max temperature.")

# Step 7: Advanced Analysis
if all(col in df.columns for col in ['MinTemp', 'MaxTemp', 'Rainfall']):
    X = df[['MinTemp', 'MaxTemp']]
    y = df['Rainfall']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions and MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error for Rainfall Prediction: {mse}')
else:
    print("Error: Columns ['MinTemp', 'MaxTemp', 'Rainfall'] are required for prediction.")

# Step 8: Conclusions and Insights
if 'Rainfall' in df.columns and 'Month' in df.columns:
    monthly_rainfall = df.groupby('Month')['Rainfall'].sum()
    highest_rainfall_month = monthly_rainfall.idxmax()
    lowest_rainfall_month = monthly_rainfall.idxmin()
    print(f'Highest rainfall month: {highest_rainfall_month}, Lowest rainfall month: {lowest_rainfall_month}')
else:
    print("Error: Columns 'Rainfall' and 'Month' are required for insights.")
