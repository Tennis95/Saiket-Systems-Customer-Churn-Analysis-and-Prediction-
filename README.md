 # Customer-Churn-Analysis-and-Prediction
The project aims to analyze customer churn in a telecommunications company and develop predictive models to identify at-risk customers. The ultimate goal is to provide actionable insights and recommendations to reduce churn and improve customer retention.

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.feature_selection import RFE

# Load the dataset
data = pd.read_csv('/content/Telco_Customer_Churn_Dataset  (3).csv’)

print(data.head())

Data Preparation

In this task, you will be responsible for loading the dataset and conducting an initial exploration. Handle missing values, and if necessary, convert categorical variables into numerical representations. Furthermore, split the dataset into training and testing sets for subsequent model evaluation.

# Fill missing TotalCharges with 0
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)

# Convert categorical variables to numerical representations using one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Split the dataset into training and testing sets
X = data_encoded.drop('Churn_Yes', axis=1)  # Assuming 'Churn_Yes' is the target variable after encoding
y = data_encoded['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}”)

Exploratory Data Analysis (EDA)


Calculate and visually represent the overall churn rate. Explore customer distribution by gender, partner status, and dependent status. Analyze tenure distribution and its relation with churn. Investigate how churn varies across different contract types and payment methods.

# Calculate the overall churn rate
churn_rate = data['Churn'].value_counts(normalize=True)['Yes'] * 100
print(f"Overall Churn Rate: {churn_rate:.2f}%”)

# Plot the overall churn rate
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='Churn')
plt.title('Overall Churn Distribution')
plt.show()

# Explore customer distribution by gender, partner status, and dependent status
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.countplot(data=data, x='gender', ax=axes[0])
axes[0].set_title('Customer Distribution by Gender')
sns.countplot(data=data, x='Partner', ax=axes[1])
axes[1].set_title('Customer Distribution by Partner Status')
sns.countplot(data=data, x='Dependents', ax=axes[2])
axes[2].set_title('Customer Distribution by Dependent Status')
plt.show()

# Analyze tenure distribution and its relation with churn
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='tenure', hue='Churn', multiple='stack', bins=30)
plt.title('Tenure Distribution and its Relation with Churn')
plt.show()

# Investigate how churn varies across different contract types and payment methods
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.countplot(data=data, x='Contract', hue='Churn', ax=axes[0])
axes[0].set_title('Churn by Contract Type')
sns.countplot(data=data, x='PaymentMethod', hue='Churn', ax=axes[1])
axes[1].set_title('Churn by Payment Method')
plt.xticks(rotation=45)
plt.show()

Customer Segmentation


Segment customers based on tenure, monthly charges, and contract type. Analyze churn rates within these segments. Identify high-value customers who are at risk of churning and might need special attention.

# Segment customers based on tenure, monthly charges, and contract type
bins_tenure = [0, 12, 24, 36, 48, 60, 72]
labels_tenure = ['0-12', '12-24', '24-36', '36-48', '48-60', '60-72']
data['tenure_group'] = pd.cut(data['tenure'], bins=bins_tenure, labels=labels_tenure, right=False)

bins_monthly_charges = [0, 20, 40, 60, 80, 100, 120]
labels_monthly_charges = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-120']
data['monthly_charges_group'] = pd.cut(data['MonthlyCharges'], bins=bins_monthly_charges, labels=labels_monthly_charges, right=False)

# Analyze churn rates within these segments
segment_analysis = data.groupby(['tenure_group', 'monthly_charges_group', 'Contract'])['Churn'].value_counts(normalize=True).unstack().fillna(0)
segment_analysis = segment_analysis.rename(columns={'Yes': 'Churn_Rate', 'No': 'Non_Churn_Rate'})

print(segment_analysis)

# Plot churn rates within segments
plt.figure(figsize=(14, 8))
sns.heatmap(segment_analysis['Churn_Rate'].unstack(level=2), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Churn Rate by Tenure Group, Monthly Charges Group, and Contract Type')
plt.show()
# Identify high-value customers who are at risk of churning
high_value_customers = data[(data['MonthlyCharges'] > 80) & (data['tenure'] > 12) & (data['Churn'] == 'Yes')]

# Print high-value customers at risk of churning
print("High-Value Customers at Risk of Churning:")
print(high_value_customers[['customerID', 'MonthlyCharges', 'tenure', 'Contract']])

Model Evaluation and Interpretation


Evaluate the best predictive model using the testing dataset. Interpret model coefficients or feature importances to comprehend factors influencing churn. Create ROC curves and calculate AUC for model performance assessment.

# Assuming X_train and y_train are defined somewhere earlier in your code
best_dec_tree = DecisionTreeClassifier()
best_dec_tree.fit(X_train, y_train) 

# Evaluate the best predictive model (Tuned Decision Tree) using the testing dataset
accuracy, precision, recall, f1, report = evaluate_model(best_dec_tree, X_test, y_test)
print(f"Tuned Decision Tree - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")
print("Classification Report:\n", report)

# Interpret feature importances
feature_importances = best_dec_tree.feature_importances_
indices = np.argsort(feature_importances)[::-1]
important_features = X.columns[indices]

print("Feature Importances:")
for f in range(X.shape[1]):
    print(f"{f + 1}. Feature '{important_features[f]}' ({feature_importances[indices[f]]:.4f})")


# Plot feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X.shape[1]), important_features, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Create ROC curve and calculate AUC
y_pred_prob = best_dec_tree.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

Business Recommendations


Based on the analysis and predictive models, provide actionable recommendations to the business. Suggest specific marketing strategies, retention offers, or customer engagement tactics. Estimate the potential impact of these recommendations on revenue and churn rate.

# Convert categorical variables to numerical representations using one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Split the dataset into features (X) and target variable (y)
X = data_encoded.drop('Churn_Yes', axis=1)  # Assuming 'Churn_Yes' is the target variable after encoding
y = data_encoded['Churn_Yes']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the best model (Decision Tree with best parameters)
best_dec_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=2, random_state=42)
best_dec_tree.fit(X_train, y_train)

# Feature importances
feature_importances = best_dec_tree.feature_importances_
indices = np.argsort(feature_importances)[::-1]
important_features = X.columns[indices]

# Display the most important features
print("Top 10 Features Influencing Churn:")
for i in range(10):
 print(f"{i + 1}. Feature '{important_features[i]}' ({feature_importances[indices[i]]:.4f})”)

# Identify high-risk customer segments
high_risk_customers = data[(data['MonthlyCharges'] > 70) & (data['Contract'] == 'Month-to-month') & (data['Churn'] == 'Yes')]

# Provide recommendations based on high-risk segments
print("\nRecommendations:")
print("1. Offer personalized discounts or promotions to high-risk customers with high monthly charges and month-to-month contracts.")
print("2. Enhance customer engagement through personalized communication, addressing specific needs and concerns.")
print("3. Provide loyalty rewards for long-term customers to encourage them to switch to longer-term contracts.")
print("4. Improve service quality and customer support to address issues promptly and reduce churn.")

# Estimate potential impact
average_monthly_charge = data['MonthlyCharges'].mean()
potential_revenue_loss = high_risk_customers.shape[0] * average_monthly_charge
print(f"\nEstimated Potential Revenue Loss if high-risk customers churn: ${potential_revenue_loss:.2f} per month")

# Assuming a reduction in churn rate by 10% through retention strategies
estimated_reduction = high_risk_customers.shape[0] * 0.10
reduced_churn_revenue_impact = estimated_reduction * average_monthly_charge
print(f"Estimated Revenue Saved by reducing churn by 10%: ${reduced_churn_revenue_impact:.2f} per month”)

——- END -----
