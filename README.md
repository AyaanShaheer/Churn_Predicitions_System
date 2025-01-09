# Churn_Predicition_System
# Customer Churn Prediction System

## Overview
This project implements a machine learning system to predict customer churn for telecom services. It analyzes customer behavior patterns and identifies key factors that contribute to customer churn, helping businesses take proactive measures to retain customers.

## Features
- **Data Preprocessing**: Handles missing values, categorical variables, and feature scaling
- **Exploratory Data Analysis**: Comprehensive visualizations of customer patterns
- **Feature Engineering**: Creates relevant features from raw customer data
- **Machine Learning Model**: Random Forest Classifier for churn prediction
- **Performance Metrics**: Detailed evaluation of model performance
- **Visualization Suite**: Multiple visualizations for understanding churn patterns

## Technologies Used
- Python 3.x
- Libraries:
  - pandas: Data manipulation and analysis
  - numpy: Numerical computations
  - scikit-learn: Machine learning algorithms
  - seaborn: Statistical data visualization
  - matplotlib: Creating static visualizations

## Installation

1. Clone the repository
```bash
git clone [repository-url]
cd customer-churn-prediction
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## Usage

1. Basic usage of the ChurnPredictionSystem:
```python
from src.churn_prediction_system import ChurnPredictionSystem

# Initialize the system
churn_system = ChurnPredictionSystem()

# Load and preprocess data
data = churn_system.load_telco_data("path_to_data.csv")
processed_data = churn_system.preprocess_telco_data(data)

# Train model and make predictions
feature_importance = churn_system.train_model(X_train, y_train)
predictions, probabilities = churn_system.predict_churn(X_test)
```

2. Running visualizations:
```python
# Create visualizations
churn_system.visualize_results(processed_data)
```

## Model Performance
Our current model achieves:
- Overall accuracy: 79%
- Precision for non-churners: 83%
- Recall for non-churners: 91%
- F1-score: 0.78 (weighted average)

## Key Findings
1. Month-to-month contracts have the highest churn rate
2. Fiber optic service customers are more likely to churn
3. Electronic check payment method correlates with higher churn
4. Longer tenure correlates with lower churn probability

## Data Requirements
The system expects a CSV file with the following columns:
- customerID
- tenure
- Contract
- MonthlyCharges
- TotalCharges
- [Other telecom service features]

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make changes and commit (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Future Improvements
- [ ] Implement more advanced feature engineering
- [ ] Add support for different types of models
- [ ] Create an interactive dashboard
- [ ] Add cross-validation and hyperparameter tuning
- [ ] Implement real-time prediction capabilities

## License
This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments
- Dataset provided by IBM Sample Data Sets
- Inspired by real-world telecom industry challenges

## Contact
Your Name - [your.email@example.com]
Project Link: [https://github.com/yourusername/customer-churn-prediction]
