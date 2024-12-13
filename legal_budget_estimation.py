import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Data Sources
# Internal Data (Mock Data for Illustration)
data = {
    'country': ['A', 'A', 'B', 'B', 'C', 'C'],
    'product': ['SkinCare', 'Snack', 'SkinCare', 'Snack', 'SkinCare', 'Snack'],
    'regulatory_complexity': [0.9, 0.4, 0.7, 0.3, 0.6, 0.2],
    'sentiment_score': [0.2, 0.8, 0.4, 0.9, 0.5, 0.7],
    'esg_score': [0.3, 0.5, 0.6, 0.8, 0.7, 0.9],
    'historical_cost': [100000, 50000, 120000, 40000, 90000, 30000],
    'litigation_count': [10, 2, 15, 1, 12, 0]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# 2. Feature-Specific Data
# Mock Data for Consumer Sentiment
consumer_sentiment_data = {
    'country': ['A', 'B', 'C'],
    'product': ['SkinCare', 'Snack', 'SkinCare'],
    'sentiment_score': [0.8, 0.6, 0.7]  # Aggregated consumer sentiment
}
consumer_sentiment_df = pd.DataFrame(consumer_sentiment_data)

# Mock Data for Regulatory Trends
regulatory_trends_data = {
    'country': ['A', 'B', 'C'],
    'product': ['SkinCare', 'Snack', 'SkinCare'],
    'regulatory_complexity': [0.9, 0.4, 0.6]  # Complexity scores based on regulations
}
regulatory_trends_df = pd.DataFrame(regulatory_trends_data)

# Mock Data for ESG Scores
esg_data = {
    'country': ['A', 'B', 'C'],
    'product': ['SkinCare', 'Snack', 'SkinCare'],
    'esg_score': [0.7, 0.5, 0.8]  # ESG scores for each product-country combination
}
esg_df = pd.DataFrame(esg_data)

# Mock Data for Historical Litigation
litigation_data = {
    'country': ['A', 'B', 'C'],
    'product': ['SkinCare', 'Snack', 'SkinCare'],
    'litigation_count': [5, 2, 3],
    'historical_cost': [120000, 40000, 90000]
}
litigation_df = pd.DataFrame(litigation_data)

# Merging All Data Sources
merged_df = pd.merge(consumer_sentiment_df, regulatory_trends_df, on=['country', 'product'])
merged_df = pd.merge(merged_df, esg_df, on=['country', 'product'])
merged_df = pd.merge(merged_df, litigation_df, on=['country', 'product'])

# 3. Feature Engineering
merged_df['risk_score'] = (
    merged_df['regulatory_complexity'] * 0.5 +
    merged_df['sentiment_score'] * 0.3 +
    merged_df['esg_score'] * 0.2
)

# Encode categorical variables
def encode_categorical(df):
    df_encoded = pd.get_dummies(df, columns=['country', 'product'], drop_first=True)
    return df_encoded

encoded_df = encode_categorical(merged_df)

# Splitting data for classification and regression
X = encoded_df.drop(['litigation_count', 'historical_cost'], axis=1)
y_class = (merged_df['litigation_count'] > 3).astype(int)  # Binary classification for high litigation risk
y_reg = merged_df['historical_cost']

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_class = scaler.fit_transform(X_train_class)
X_test_class = scaler.transform(X_test_class)
X_train_reg = scaler.fit_transform(X_train_reg)
X_test_reg = scaler.transform(X_test_reg)

# 4. Methodology
# Step 1: Risk Prediction Model
class_model = GradientBoostingClassifier(random_state=42)
class_model.fit(X_train_class, y_train_class)
y_pred_class = class_model.predict(X_test_class)

# Step 2: Cost Prediction Model
reg_model = GradientBoostingRegressor(random_state=42)
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

# Evaluation
class_accuracy = accuracy_score(y_test_class, y_pred_class)
reg_mse = mean_squared_error(y_test_reg, y_pred_reg)

# Output results
print("Classification Accuracy (High Litigation Risk):", class_accuracy)
print("Regression Mean Squared Error (Legal Costs):", reg_mse)

# Mock Prediction for New Data
new_data = {
    'country': ['A'],
    'product': ['SkinCare'],
    'sentiment_score': [0.8],
    'regulatory_complexity': [0.9],
    'esg_score': [0.7]
}
new_df = pd.DataFrame(new_data)
new_df['risk_score'] = new_df['regulatory_complexity'] * 0.5 + new_df['sentiment_score'] * 0.3 + new_df['esg_score'] * 0.2
new_df_encoded = encode_categorical(new_df)

# Align columns with training data
new_df_encoded = new_df_encoded.reindex(columns=X.columns, fill_value=0)
new_scaled = scaler.transform(new_df_encoded)

litigation_risk_pred = class_model.predict(new_scaled)
legal_cost_pred = reg_model.predict(new_scaled)

print("Predicted High Litigation Risk:", litigation_risk_pred)
print("Predicted Legal Cost:", legal_cost_pred)

# Monte Carlo Simulation for Uncertainty Adjustment
def monte_carlo_simulation(model, X, n_simulations=1000):
    simulations = []
    for _ in range(n_simulations):
        # Add random noise to simulate uncertainty
        noise = np.random.normal(0, 0.1, X.shape)
        X_noisy = X + noise
        predictions = model.predict(X_noisy)
        simulations.append(predictions)
    
    simulations = np.array(simulations)
    return simulations

# Apply Monte Carlo Simulation to Cost Predictions
n_simulations = 1000
cost_simulations = monte_carlo_simulation(reg_model, X_test_reg, n_simulations=n_simulations)

# Calculate Confidence Intervals
lower_bound = np.percentile(cost_simulations, 2.5, axis=0)
upper_bound = np.percentile(cost_simulations, 97.5, axis=0)

print("95% Confidence Interval for Cost Predictions:")
for i, (lb, ub) in enumerate(zip(lower_bound, upper_bound)):
    print(f"Sample {i + 1}: {lb:.2f} - {ub:.2f}")

# Visualization of Uncertainty
def plot_uncertainty(predictions, lower, upper):
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label="Predicted Costs", marker='o')
    plt.fill_between(range(len(predictions)), lower, upper, color='b', alpha=0.2, label="95% Confidence Interval")
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Cost")
    plt.title("Prediction Uncertainty via Monte Carlo Simulation")
    plt.legend()
    plt.show()

plot_uncertainty(y_pred_reg, lower_bound, upper_bound)

# Final Budget Aggregation by Country
merged_df['predicted_cost'] = reg_model.predict(scaler.transform(encode_categorical(merged_df.drop(['litigation_count', 'historical_cost'], axis=1))))

# Aggregate predicted costs by country
country_budget = merged_df.groupby('country')['predicted_cost'].sum().reset_index()
print("Aggregated Budget by Country:")
print(country_budget)

# Visualizing Budget by Country
plt.figure(figsize=(8, 6))
plt.bar(country_budget['country'], country_budget['predicted_cost'], color='skyblue')
plt.xlabel('Country')
plt.ylabel('Total Predicted Legal Budget')
plt.title('Aggregated Legal Budget by Country')
plt.show()
