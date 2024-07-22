import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv(r"C:\Users\siddh\OneDrive\Desktop\codsoft intern\Titanic\Titanic-Dataset.csv")

# Preprocess the data
def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Convert 'Sex' to numerical
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Convert 'Embarked' to numerical
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Fill missing values
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df

# Preprocess the data
df_processed = preprocess_data(df)

# Split features and target
X = df_processed.drop('Survived', axis=1)
y = df_processed['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Function to get user input
def get_user_input():
    print("\nEnter passenger information:")
    pclass = int(input("Passenger Class (1, 2, or 3): "))
    sex = input("Sex (male or female): ")
    age = float(input("Age: "))
    sibsp = int(input("Number of Siblings/Spouses Aboard: "))
    parch = int(input("Number of Parents/Children Aboard: "))
    fare = float(input("Fare: "))
    embarked = input("Port of Embarkation (S, C, or Q): ")
    
    # Convert categorical inputs
    sex = 0 if sex.lower() == 'male' else 1
    embarked = {'s': 0, 'c': 1, 'q': 2}[embarked.lower()]
    
    return [pclass, sex, age, sibsp, parch, fare, embarked]

# Function to make prediction
def predict_survival(passenger_data):
    passenger_df = pd.DataFrame([passenger_data], columns=X.columns)
    passenger_scaled = scaler.transform(passenger_df)
    prediction = model.predict(passenger_scaled)
    probability = model.predict_proba(passenger_scaled)[0][1]
    return prediction[0], probability

# Main loop
while True:
    user_input = get_user_input()
    prediction, probability = predict_survival(user_input)
    
    print("\nSurvival Prediction:")
    if prediction == 1:
        print(f"The passenger would likely SURVIVE with a probability of {probability:.2f}")
    else:
        print(f"The passenger would likely NOT SURVIVE with a probability of {1-probability:.2f}")
    
    again = input("\nWould you like to make another prediction? (yes/no): ")
    if again.lower() != 'yes':
        break

print("Thank you for using the Titanic Survival Prediction model!")