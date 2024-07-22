import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\siddh\OneDrive\Desktop\codsoft intern\Titanic\Titanic-Dataset.csv")
    return df

# Preprocess the data
def preprocess_data(df):
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df

# Train the model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Load and preprocess data
df = load_data()
df_processed = preprocess_data(df)
X = df_processed.drop('Survived', axis=1)
y = df_processed['Survived']

# Train model
model, scaler = train_model(X, y)

# Streamlit app
st.title("Titanic Survival Prediction")

# Sidebar for user input
st.sidebar.header("Passenger Information")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.radio("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 0, 100, 30)
sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.number_input("Fare", 0.0, 600.0, 32.2)
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Convert user input to model input
sex = 0 if sex == "Male" else 1
embarked = {"S": 0, "C": 1, "Q": 2}[embarked]
user_input = [pclass, sex, age, sibsp, parch, fare, embarked]

# Make prediction
if st.sidebar.button("Predict Survival"):
    user_input_scaled = scaler.transform([user_input])
    prediction = model.predict(user_input_scaled)
    probability = model.predict_proba(user_input_scaled)[0][1]
    
    st.header("Prediction Result")
    if prediction[0] == 1:
        st.success(f"The passenger would likely SURVIVE with a probability of {probability:.2f}")
    else:
        st.error(f"The passenger would likely NOT SURVIVE with a probability of {1-probability:.2f}")

# Display dataset information
st.header("Dataset Information")
st.write(df.describe())

# Visualizations
st.header("Data Visualizations")

# Survival by Passenger Class
fig, ax = plt.subplots()
df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=ax)
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival Rate")
ax.set_title("Survival Rate by Passenger Class")
st.pyplot(fig)

# Survival by Sex
fig, ax = plt.subplots()
df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=ax)
ax.set_xlabel("Sex")
ax.set_ylabel("Survival Rate")
ax.set_title("Survival Rate by Sex")
st.pyplot(fig)

# Age distribution
fig, ax = plt.subplots()
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', ax=ax)
ax.set_title("Age Distribution by Survival")
st.pyplot(fig)

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

st.header("Feature Importance")
fig, ax = plt.subplots()
sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
ax.set_title("Feature Importance in Survival Prediction")
st.pyplot(fig)