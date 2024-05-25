import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
df=pd.read_csv(r"./drugs_side_effects_drugs_com.csv")
df.drop(columns=["csa", "pregnancy_category", "rx_otc", "alcohol", "activity", "drug_classes"], inplace=True)
col = df.columns
print(df)

missing_values = df.isnull().sum()
for i in range(len(col)):
    print(col[i] + ": " + str(missing_values[i]))
df.dropna(inplace=True)
missing_values = df.isnull().sum()
for i in range(len(col)):
    print(col[i] + ": " + str(missing_values[i]))
duplicates = df[df.duplicated()]
print("Duplicate Rows:")
print(duplicates)
duplicates = df[df.duplicated(subset=['drug_name'])]
print("Duplicate Rows based on 'drug_name':")
print(duplicates)
df['drug_name'] = df['drug_name'].str.lower().str.strip()
df['generic_name'] = df['generic_name'].str.lower().str.strip()

def find_least_side_effects_drug(drug_name):
    # Normalize the input drug name
    drug_name = drug_name.lower().strip()

    # Debug: Print the normalized input drug name
    print(f"Normalized input drug name: '{drug_name}'")

    # Check if the drug name exists in the dataset
    if drug_name not in df['drug_name'].values:
        return "Drug name not found in the dataset."

    # Get the generic name of the input drug
    generic_name = df.loc[df['drug_name'] == drug_name, 'generic_name'].values[0]

    # Debug: Print the generic name found
    print(f"Generic name for '{drug_name}': '{generic_name}'")

    # Get all drugs with the same generic name
    similar_drugs = df[df['generic_name'] == generic_name].copy()

    # Debug: Print similar drugs
    print("Similar drugs:\n", similar_drugs)

    # Count the side effects for each drug
    similar_drugs['side_effect_count'] = similar_drugs['side_effects'].apply(
        lambda x: len(str(x).split(',')) if pd.notnull(x) and x.strip() != '' else 0
    )

    # Debug: Print side effect counts
    print("Side effect counts:\n", similar_drugs[['drug_name', 'side_effect_count']])

    # Find the drug with the fewest side effects
    least_side_effects_drug = similar_drugs.loc[similar_drugs['side_effect_count'].idxmin()]['drug_name']

    return least_side_effects_drug

# Example usage
user_drug_name = 'fenfluramine'  # Replace with the actual drug name entered by the user
result = find_least_side_effects_drug(user_drug_name)
print(f"The drug with the least side effects is: {result}")
df2 = df[['drug_name']]

df2['in_stock'] = np.random.randint(2, size=df2.shape[0])
def check_stock(drug_name):
    stock_status = df2[df2['drug_name'] == drug_name]['in_stock'].values
    if len(stock_status) == 0:
        return 0
    else:
        return 1

df2
import pandas as pd
import numpy as np

# Generate synthetic data for df2
num_rows = 1000
current_stock_level = np.random.randint(1, 1000, size=num_rows)
sales_volume = np.random.randint(1, 100, size=num_rows)
reorder_point = np.random.randint(10, 100, size=num_rows)
lead_time = np.random.randint(1, 30, size=num_rows)
supplier_lead_time_variability = np.random.uniform(0.1, 1.0, size=num_rows)
demand_forecast = np.random.randint(50, 200, size=num_rows)
expiration_date = np.random.choice(pd.date_range('2022-01-01', '2023-12-31'), size=num_rows)
seasonality = np.random.choice(['Low', 'Medium', 'High'], size=num_rows)
special_events = np.random.choice([True, False], size=num_rows)
storage_conditions = np.random.choice(['Dry', 'Refrigerated', 'Frozen'], size=num_rows)
profit_margin = np.random.uniform(0.1, 0.5, size=num_rows)
competition = np.random.choice(['Low', 'Medium', 'High'], size=num_rows)
trends = np.random.choice(['Increasing', 'Decreasing', 'Stable'], size=num_rows)

# Generate synthetic values for Restocking_Needed column (randomly)
restocking_needed = np.random.choice([0, 1], size=num_rows)

# Create DataFrame df2
data = {
    'Current_Stock_Level': current_stock_level,
    'Sales_Volume': sales_volume,
    'Reorder_Point': reorder_point,
    'Lead_Time': lead_time,
    'Supplier_Lead_Time_Variability': supplier_lead_time_variability,
    'Demand_Forecast': demand_forecast,
    'Expiration_Date': expiration_date,
    'Seasonality': seasonality,
    'Special_Events': special_events,
    'Storage_Conditions': storage_conditions,
    'Profit_Margin': profit_margin,
    'Competition': competition,
    'Trends': trends,
    'Restocking_Needed': restocking_needed  # Add Restocking_Needed column
}
df2 = pd.DataFrame(data)

# Load the dataset containing medicine names
medicine_names_df = pd.read_csv("./drugs_side_effects_drugs_com.csv")  # Assuming the column name is "Medicine_Name"

# Select 1000 random medicine names from the loaded dataset
selected_medicine_names = np.random.choice(medicine_names_df['drug_name'], num_rows)

# Add the column of medicine names to df2
df2['drug_name'] = selected_medicine_names

df2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier



# Define features and target variable
X = df2[['Current_Stock_Level', 'Sales_Volume', 'Reorder_Point', 'Lead_Time', 'Supplier_Lead_Time_Variability',
        'Demand_Forecast', 'Profit_Margin', 'Seasonality', 'Special_Events', 'Storage_Conditions',
        'Competition', 'Trends']]
y = df2['Restocking_Needed']

# Convert categorical features to dummy variables
X = pd.get_dummies(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize lists to store training and test accuracies
training_accuracy = []
test_accuracy = []

# Define range of neighbors (k)
neighbors_settings = range(1, 11)

# Loop over different values of k
for n_neighbors in neighbors_settings:
    # Initialize and train the KNN classifier
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train_scaled, y_train)

    # Calculate training and test accuracies
    training_accuracy.append(clf.score(X_train_scaled, y_train))
    test_accuracy.append(clf.score(X_test_scaled, y_test))

def train_knn_model():
    X = df2[['Current_Stock_Level', 'Sales_Volume', 'Reorder_Point', 'Lead_Time', 'Supplier_Lead_Time_Variability',
             'Demand_Forecast', 'Profit_Margin', 'Seasonality', 'Special_Events', 'Storage_Conditions',
             'Competition', 'Trends']]
    y = df2['Restocking_Needed']
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    k = 5
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_scaled, y_train)
    y_pred = knn_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Train the KNN model upon initialization
train_knn_model()