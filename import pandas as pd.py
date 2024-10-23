import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
df = pd.read_csv ('C:\\Users\\mavs2\\skilltrack_project\\AI & ML project\\user_behavior_dataset.csv') 

# Display the first few rows of the DataFrame
print(df.head())

# Display the structure of the DataFrame
print(df.info())

# Display basic statistics of the DataFrame
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Display the columns in the DataFrame
print(df.columns)

# Histogram of App Usage Time
plt.figure(figsize=(10, 6))
sns.histplot(df['App Usage Time (min/day)'], bins=30)
plt.title('Distribution of App Usage Time')
plt.xlabel('App Usage Time (min/day)')
plt.ylabel('Frequency')
plt.show()

# Correlation Heatmap
# Select only numerical columns
numerical_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(12, 8))
sns.heatmap(numerical_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



# Box Plot for App Usage Time by Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='App Usage Time (min/day)', data=df)
plt.title('App Usage Time by Gender')
plt.xlabel('Gender')
plt.ylabel('App Usage Time (min/day)')
plt.show()

# Count Plot for Device Model
plt.figure(figsize=(12, 6))
sns.countplot(y='Device Model', data=df)
plt.title('Count of Users by Device Model')
plt.xlabel('Count')
plt.ylabel('Device Model')
plt.show()

#Distribution  of app usage time
plt.figure(figsize=(10, 6))
sns.histplot(df['App Usage Time (min/day)'], bins=30, kde=True)
plt.title('Distribution of App Usage Time (min/day)')
plt.xlabel('App Usage Time (min/day)')
plt.ylabel('Frequency')
plt.show()

#Box Plot of App Usage Time by Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='App Usage Time (min/day)', data=df)
plt.title('App Usage Time by Gender')
plt.xlabel('Gender')
plt.ylabel('App Usage Time (min/day)')
plt.show()

#Count Plot of Device Model
plt.figure(figsize=(12, 6))
sns.countplot(y='Device Model', data=df, order=df['Device Model'].value_counts().index)
plt.title('Count of Users by Device Model')
plt.xlabel('Count')
plt.ylabel('Device Model')
plt.show()


#Heatmap of Correlation Matrix
plt.figure(figsize=(10, 8))
# Select only the numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#Statistical analysis
#Descriptive Analysis
print("Descriptive Statistic:")
print(df.describe(include='all'))

# Group Analysis: App Usage Time by Gender
grouped_gender = df.groupby('Gender')[['App Usage Time (min/day)', 'Age', 'Data Usage (MB/day)']].mean()
print("\nAverage App Usage Time, Age, and Data Usage by Gender:")
print(grouped_gender)

# Group Analysis: App Usage Time by Operating System
grouped_os = df.groupby('Operating System')[['App Usage Time (min/day)', 'Age', 'Data Usage (MB/day)']].mean()
print("\nAverage App Usage Time, Age, and Data Usage by Operating System:")
print(grouped_os)


# Correlation Analysis
correlation_age_usage = df['Age'].corr(df['App Usage Time (min/day)'])
print(f"\nCorrelation between Age and App Usage Time: {correlation_age_usage:.2f}")

# Correlation between Data Usage and App Usage Time
correlation_data_usage = df['Data Usage (MB/day)'].corr(df['App Usage Time (min/day)'])
print(f"Correlation between Data Usage and App Usage Time: {correlation_data_usage:.2f}")

#Training the ML model
# Importing required libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Features and target variable
X = df[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)',
        'Number of Apps Installed', 'Data Usage (MB/day)', 'Age']]  # Features
y = df['User Behavior Class']  # Target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Decision Tree Classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

#Data Scaling
from sklearn.preprocessing import StandardScaler

# Select features for scaling (excluding the target variable)
X = df.drop(columns=['User Behavior Class'])  # Replace with your feature columns
y = df['User Behavior Class']  # Target variable

# Encode categorical variables if necessary
X = pd.get_dummies(X, columns=['Device Model', 'Operating System', 'Gender'], drop_first=True)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

# Define the model
rf = RandomForestClassifier()

# Define the hyperparameter grid
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Set up the randomized search with cross-validation
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the model
rf_random.fit(X_train, y_train)

# Get the best parameters
print("Best Parameters: ", rf_random.best_params_)


#Evaluate the best model
best_rf = rf_random.best_estimator_

# Predict and evaluate
y_pred = best_rf.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Random Forest Model Accuracy: {accuracy}")
