# House Price Prediction Model

A machine learning model that predicts house prices based on property features using linear regression. This project demonstrates fundamental ML concepts including data preprocessing, model training, evaluation, and deployment.

## ⚠️ Development Status

**This project is currently under development.** The model is trained on synthetic, human-generated data for demonstration and testing purposes. Real-world deployment would require actual market data from reliable sources.

## Features

- **Linear Regression Model** for price prediction
- **Categorical Data Handling** with one-hot encoding for location
- **Model Persistence** with joblib serialization
- **Preprocessing Pipeline** for consistent data transformation
- **Performance Evaluation** with Mean Squared Error metrics
- **Easy Prediction Interface** for new property data

## Dataset

### Current Data (Demo/Training)
The model currently uses **synthetic, human-generated data** for development purposes:
- **20 sample properties** with varying characteristics
- **Features**: Size (sq ft), Location (urban/suburb/rural), Bedrooms
- **Target**: Price (USD)

### Data Structure
```csv
size,location,bedrooms,price
1200,urban,2,250000
1500,suburb,3,320000
1800,urban,3,380000
...
```

**Note**: This is demonstration data. For production use, replace with real market data from MLS, Zillow API, or other reliable real estate sources.

## Requirements

### Dependencies

```bash
pip install pandas scikit-learn joblib numpy
```

### Python Version
- Python 3.7 or higher

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the dataset file:
   ```
   house_prices.csv
   ```

## Usage

### Training the Model

Run the main script to train and save the model:

```bash
python house_price_predictor.py
```

This will:
1. Load the dataset
2. Preprocess the data (one-hot encode locations)
3. Split data into training/testing sets
4. Train the linear regression model
5. Evaluate performance
6. Save the trained model and preprocessor

### Making Predictions

After training, you can make predictions on new data:

```python
import pandas as pd
import joblib

# Load the saved model and preprocessor
saved_data = joblib.load('house_price_model.pkl')
model = saved_data['model']
preprocessor = saved_data['preprocessor']

# Prepare new data
new_house = pd.DataFrame({
    'size': [1400],
    'location': ['suburb'],
    'bedrooms': [3]
})

# Make prediction
processed_data = preprocessor.transform(new_house)
predicted_price = model.predict(processed_data)
print(f'Predicted price: ${predicted_price[0]:,.2f}')
```

## Model Details

### Algorithm
- **Linear Regression**: Simple, interpretable model suitable for continuous target variables
- **Regularization**: None (basic linear regression)
- **Features**: 3 input features (size, location, bedrooms)

### Preprocessing Pipeline
1. **One-Hot Encoding**: Converts categorical 'location' feature to binary columns
2. **Feature Scaling**: Not applied (linear regression can handle different scales)
3. **Column Transformation**: Maintains numerical features while encoding categorical ones

### Model Performance
Current performance metrics on demo data:
- **Mean Squared Error**: Varies based on train/test split
- **R² Score**: Available for model evaluation

**Note**: Performance metrics are based on synthetic data and may not reflect real-world accuracy.

## File Structure

```
house-price-prediction/
│
├── house_price_predictor.py    # Main training script
├── house_prices.csv           # Dataset (demo data)
├── house_price_model.pkl      # Saved model and preprocessor
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── examples/
    └── predict_example.py     # Example prediction script
```

## Features Explained

### Input Features

1. **Size** (numerical)
   - House size in square feet
   - Range: 950 - 2400 sq ft (in demo data)

2. **Location** (categorical)
   - Property location type
   - Categories: 'urban', 'suburb', 'rural'
   - One-hot encoded during preprocessing

3. **Bedrooms** (numerical)
   - Number of bedrooms
   - Range: 1 - 5 bedrooms (in demo data)

### Target Variable

- **Price** (numerical)
  - House price in USD
  - Range: $180,000 - $470,000 (in demo data)

## Data Preprocessing

### One-Hot Encoding
The 'location' categorical feature is transformed:
```
location_rural, location_suburb, location_urban
     0,              1,              0         # suburb
     0,              0,              1         # urban
     1,              0,              0         # rural
```

### Pipeline Benefits
- **Consistency**: Same preprocessing applied to training and prediction data
- **Persistence**: Preprocessor saved with model for future use
- **Scalability**: Easy to add new preprocessing steps

## Model Evaluation

### Current Metrics
- **Mean Squared Error (MSE)**: Measures average squared difference between predicted and actual prices
- **Train/Test Split**: 80/20 split with random_state=42 for reproducibility

### Evaluation Process
1. Split data into training and testing sets
2. Train model on training data
3. Evaluate on held-out test data
4. Calculate MSE to assess performance

## Future Improvements

### Data Enhancements
- [ ] **Real Market Data**: Replace synthetic data with actual real estate data
- [ ] **More Features**: Add bathrooms, garage, year built, square footage of lot
- [ ] **Geographic Data**: Include ZIP codes, neighborhood ratings
- [ ] **Market Trends**: Historical price trends, seasonal adjustments

### Model Improvements
- [ ] **Feature Engineering**: Create new features from existing ones
- [ ] **Advanced Models**: Random Forest, Gradient Boosting, Neural Networks
- [ ] **Cross-Validation**: More robust model evaluation
- [ ] **Hyperparameter Tuning**: Optimize model parameters
- [ ] **Feature Selection**: Identify most important features

### Technical Enhancements
- [ ] **API Development**: REST API for predictions
- [ ] **Web Interface**: User-friendly prediction interface
- [ ] **Data Validation**: Input validation and error handling
- [ ] **Model Monitoring**: Track model performance over time
- [ ] **A/B Testing**: Compare different model versions

## Limitations

### Current Limitations
1. **Synthetic Data**: Model trained on artificial, limited dataset
2. **Simple Model**: Linear regression may not capture complex relationships
3. **Limited Features**: Only 3 features may not represent all price factors
4. **No Validation**: No cross-validation or advanced evaluation metrics
5. **Static Model**: No mechanism for retraining with new data

### Data Quality Concerns
- **Sample Size**: Only 20 data points for training
- **Geographic Scope**: No specific geographic area represented
- **Time Period**: No temporal aspect considered
- **Market Conditions**: No economic factors included

## Deployment Considerations

Before production deployment:

1. **Data Quality**: Use real, validated market data
2. **Legal Compliance**: Ensure compliance with fair housing laws
3. **Bias Testing**: Test for discriminatory patterns
4. **Performance Monitoring**: Implement model performance tracking
5. **Regular Updates**: Plan for model retraining with new data

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Add unit tests for new features
- Update documentation for any changes
- Follow PEP 8 style guidelines
- Include example usage for new functionality

## Disclaimer

This model is for **educational and demonstration purposes only**. The predictions are based on synthetic data and should not be used for actual real estate decisions. Always consult with real estate professionals and use current market data for property valuations.

## Contact

For questions, suggestions, or collaboration opportunities:
- Email: imad.ansarilol@gmail.com

## Acknowledgments

- **scikit-learn** for machine learning tools
- **pandas** for data manipulation
- **joblib** for model persistence
- Real estate professionals who inspire data-driven approaches to property valuation
