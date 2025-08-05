import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('./CVD_Dataset_Cleaned.csv')

# Encode categorical variables
categorical_columns = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 
                      'Family History of CVD', 'Blood Pressure Category']

for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Define X and y
columns_to_drop = ['CVD Risk Level', 'Blood Pressure (mmHg)']
X = df.drop(columns_to_drop, axis=1)
y = df['CVD Risk Level']

# Scale X
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of model: {accuracy * 100:.2f}%")

# Detailed report
print(classification_report(y_test, y_pred))
