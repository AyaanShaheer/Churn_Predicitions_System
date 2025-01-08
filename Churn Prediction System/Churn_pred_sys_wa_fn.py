import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(r"C:\Users\Ayaan\OneDrive\Desktop\Churn Prediction System\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Create figure for contract type analysis
plt.figure(figsize=(15, 5))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.show()

# Create figure for internet service analysis
plt.figure(figsize=(15, 5))
sns.countplot(data=df, x='InternetService', hue='Churn')
plt.title('Churn by Internet Service Type')
plt.xlabel('Internet Service Type')
plt.ylabel('Number of Customers')
plt.show()

# Create figure for payment method analysis
plt.figure(figsize=(15, 5))
sns.countplot(data=df, x='PaymentMethod', hue='Churn')
plt.title('Churn by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()

# Create correlation matrix for numeric columns
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols + ['Churn']].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numeric Features')
plt.show()

# Create multiple service analysis
services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

plt.figure(figsize=(15, 8))
churn_rates = []
service_names = []

for service in services:
    service_churn = df.groupby(service)['Churn'].mean()
    for category, rate in service_churn.items():
        churn_rates.append(rate * 100)
        service_names.append(f'{service}\n({category})')

plt.bar(service_names, churn_rates)
plt.title('Churn Rate by Service Type and Category')
plt.xlabel('Service Type (Category)')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Tenure distribution by contract type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Contract', y='tenure', hue='Churn')
plt.title('Tenure Distribution by Contract Type and Churn Status')
plt.show()

# Monthly charges by internet service type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='InternetService', y='MonthlyCharges', hue='Churn')
plt.title('Monthly Charges by Internet Service Type and Churn Status')
plt.show()

# Additional analysis: Churn rate by tenure groups
df['tenure_group'] = pd.qcut(df['tenure'], q=4, labels=['0-25%', '25-50%', '50-75%', '75-100%'])
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='tenure_group', y='Churn')
plt.title('Churn Rate by Tenure Quartile')
plt.xlabel('Tenure Group')
plt.ylabel('Churn Rate')
plt.show()