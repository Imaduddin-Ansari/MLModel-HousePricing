import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load dataset
data = pd.read_csv('house_prices.csv')

# Define features and target
X = data[['size', 'location', 'bedrooms']]
y = data['price']

# Preprocessing: One-hot encode the 'location' column
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['location'])
    ],
    remainder='passthrough'  # keeps the other columns (size, bedrooms)
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Save both the model and preprocessor for future use
joblib.dump({
    'model': model,
    'preprocessor': preprocessor
}, 'house_price_model.pkl')

# Load for future predictions
saved_data = joblib.load('house_price_model.pkl')
loaded_model = saved_data['model']
loaded_preprocessor = saved_data['preprocessor']

# Example prediction - must preprocess new data the same way
example_data = pd.DataFrame({
    'size': [1200],
    'location': ['urban'],
    'bedrooms': [2]
})

# Preprocess the example data before prediction
example_processed = loaded_preprocessor.transform(example_data)
example_prediction = loaded_model.predict(example_processed)
print(f'Predicted house price: ${example_prediction[0]:,.2f}')