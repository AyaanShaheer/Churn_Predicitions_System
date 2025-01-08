#same but with explicitly mentioned example
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

class ChurnPredictionSystem:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def preprocess_data(self, df):
        """
        Preprocess the input data for modeling
        """
        # Create features for customer behavior
        df['tenure_months'] = pd.to_numeric(df['tenure_months'], errors='coerce')
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
        
        # Calculate additional features
        df['avg_monthly_charges'] = df['total_charges'] / df['tenure_months']
        df['charges_trend'] = df['total_charges'] / (df['tenure_months'] ** 2)
        
        # Convert categorical variables to numeric
        categorical_columns = df.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df, columns=categorical_columns)
        
        # Handle missing values
        df_encoded = pd.DataFrame(self.imputer.fit_transform(df_encoded), 
                                columns=df_encoded.columns)
        
        return df_encoded
    
    def train_model(self, X_train, y_train):
        """
        Train the churn prediction model
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict_churn(self, X_test):
        """
        Make predictions on new data
        """
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)
        
        return predictions, probabilities
    
    def evaluate_model(self, y_true, y_pred):
        """
        Evaluate model performance
        """
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        
        # Create confusion matrix visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def analyze_risk_factors(self, df, predictions, probabilities):
        """
        Analyze and identify key risk factors for churning customers
        """
        df['churn_probability'] = probabilities[:, 1]
        df['predicted_churn'] = predictions
        
        # Identify high-risk customers
        high_risk = df[df['churn_probability'] > 0.7]
        
        # Analyze key characteristics of high-risk customers
        risk_analysis = {
            'total_high_risk': len(high_risk),
            'avg_tenure': high_risk['tenure_months'].mean(),
            'avg_charges': high_risk['total_charges'].mean(),
            'top_factors': self.get_top_risk_factors(high_risk)
        }
        
        return risk_analysis
    
    def get_top_risk_factors(self, high_risk_customers):
        """
        Identify top factors contributing to churn risk
        """
        numerical_cols = high_risk_customers.select_dtypes(include=['float64', 'int64']).columns
        correlations = high_risk_customers[numerical_cols].corr()['churn_probability'].sort_values(ascending=False)
        
        return correlations

# Example usage
if __name__ == "__main__":
    # Sample data structure (replace with your actual data)
    sample_data = pd.DataFrame({
        'customer_id': range(1000),
        'tenure_months': np.random.randint(1, 72, 1000),
        'total_charges': np.random.uniform(100, 5000, 1000),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer'], 1000),
        'churn': np.random.choice([0, 1], 1000)
    })
    
    # Initialize the system
    churn_system = ChurnPredictionSystem()
    
    # Preprocess data
    processed_data = churn_system.preprocess_data(sample_data.copy())
    
    # Split features and target
    X = processed_data.drop('churn', axis=1)
    y = processed_data['churn']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model and get feature importance
    feature_importance = churn_system.train_model(X_train, y_train)
    
    # Make predictions
    predictions, probabilities = churn_system.predict_churn(X_test)
    
    # Evaluate model
    churn_system.evaluate_model(y_test, predictions)
    
    # Analyze risk factors
    risk_analysis = churn_system.analyze_risk_factors(sample_data.copy(), predictions, probabilities)