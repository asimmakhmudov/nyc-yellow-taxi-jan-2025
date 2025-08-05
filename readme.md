# NYC Yellow Taxi Data Analysis - January 2025

## Project Overview

This project analyzes NYC Yellow Taxi trip data for January 2025, with the goal of predicting taxi fare amounts (`total_amount`) using various features including trip characteristics, location data, temporal patterns, and weather conditions.

## Dataset Description

### Primary Data Sources
- **NYC Yellow Taxi Trip Data**: January 2025 trip records from NYC TLC  
  Source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Taxi Zone Lookup**: Borough and location mapping data (included with TLC data)
- **Weather Data**: Hourly weather conditions for NYC in January 2025  
  Source: [NYC Weather Jan 2025 - Kaggle](https://www.kaggle.com/datasets/asimmahmudov/nyc-weather-jan-2025)

### Target Variable
- `total_amount`: The total fare amount for each taxi trip

### Key Features
- **Trip Features**: Distance, passenger count, rate code, pickup/dropoff locations
- **Temporal Features**: Date, hour, day of week, weekend indicator, holiday indicator
- **Location Features**: Borough information
- **Weather Features**: Temperature, humidity, wind speed, cloud cover, precipitation


## Analysis Pipeline

### 0. Import Libraries
- Data manipulation: `pandas`, `numpy`
- Visualization: `matplotlib`
- Machine learning: `sklearn`
- Weather data: `meteostat`
- Date handling: `datetime`, holiday calendars

### 1. Data Import
- Load January 2025 yellow taxi trip data
- Import taxi zone lookup table for borough mapping

### 2. Data Exploration
- Examine data distributions and outliers
- Analyze fare amounts, including negative and zero values
- Investigate payment types and trip distances

### 3. Data Cleaning
- Filter out extreme fare values (< $0 or > $250)
- Handle missing values in passenger count and rate code
- Remove invalid trip records

### 4. Data Preparation
- Convert data types appropriately
- Extract temporal features (year, month, day, hour)
- Filter data to focus on January 2025
- Group data by location and time for aggregation

### 5. Benchmark Model
- **Features**: Location ID, month, day, hour (categorical)
- **Model**: Decision Tree Regressor (max_depth=10)
- **Split**: 67% training, 33% testing
- **Encoding**: One-hot encoding for categorical variables

### 6. Feature Engineering
- **Temporal Features**:
  - Day of week and weekend indicators
  - Federal holiday indicators
- **Location Features**:
  - Borough mapping from taxi zones
- **Weather Features**:
  - Temperature, humidity, wind speed
  - Cloud cover and precipitation data
  - Hourly weather conditions merged with trip data

### 7. Enhanced Model Training
- **Features**: All categorical features + weather variables
- **Model**: Decision Tree Regressor
- **Evaluation Metrics**: MAE, MSE, RMSE, R²

## Key Findings

### Data Quality Issues
- Some trips have negative or zero fare amounts
- Missing values in passenger count and rate code fields
- Outliers with extremely high fare amounts (>$250)

### Data Characteristics
- Dataset filtered to ~X records after cleaning (between $0-$250)
- Peak hours and popular pickup locations identified
- Weather patterns show seasonal January conditions

### Model Performance
The enhanced model with weather features shows improvement over the baseline model that uses only categorical location and time features.

## Technical Implementation

### Data Processing
- **Aggregation**: Trip data grouped by location and time periods
- **Feature Engineering**: Temporal, weather, and location-based features
- **Data Quality**: Outlier removal and missing value imputation

### Machine Learning
- **Algorithm**: Decision Tree Regressor
- **Validation**: Train-test split (67%-33%)
- **Features**: Mixed categorical and numerical features
- **Encoding**: One-hot encoding for categorical variables

### Weather Integration
- **Source**: Meteostat API for NYC weather data and [Kaggle dataset](https://www.kaggle.com/datasets/asimmahmudov/nyc-weather-jan-2025)
- **Temporal Matching**: Hourly weather matched to trip timestamps
- **Variables**: Temperature, humidity, wind, precipitation, cloud cover

## Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from datetime import datetime
from meteostat import Hourly, Point
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from pandas.tseries.holiday import USFederalHolidayCalendar
```

## Usage

1. **Setup Environment**: Install required packages
2. **Data Preparation**: Ensure all data files are in the `data/` directory
3. **Run Analysis**: Execute the Jupyter notebook `taxi-data.ipynb`
4. **Model Training**: Follow the pipeline from data cleaning through model evaluation

## Future Improvements

### Model Enhancement
- Address overfitting issues identified in current model
- Experiment with other algorithms (Random Forest, XGBoost, etc.)
- Implement cross-validation for better model assessment
- Feature selection and regularization techniques

### Feature Engineering
- Traffic data integration
- Event-based features (concerts, sports events)
- Economic indicators
- Route optimization features

### Data Expansion
- Multi-month analysis for seasonality
- Comparison with other taxi companies
- Integration with public transportation data

## Notes

- The project currently shows signs of overfitting that need to be addressed
- Weather data integration significantly improves model performance
- Borough-level analysis provides valuable geographic insights
- Holiday and weekend patterns show distinct fare characteristics

## Author

This analysis focuses on understanding fare prediction patterns in NYC's taxi system during January 2025, incorporating multiple data sources for comprehensive modeling.
