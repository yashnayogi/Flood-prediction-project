import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
df = pd.read_csv('flood_risk_dataset_india.csv')

# Rename the 'Flood Occurred' column to 'Flood Risk'
df = df.rename(columns={'Flood Occurred': 'Flood Risk'})

# Preprocess the data
categorical_cols = ['Land Cover', 'Soil Type', 'Infrastructure']  # Assuming these are categorical columns
numerical_cols = [col for col in df.columns if col not in categorical_cols + ['Flood Risk']]

# One-hot encode categorical columns
one_hot_encoder = OneHotEncoder(sparse_output=False)  # Changed to sparse_output=False to avoid the TypeError
encoded_categorical_cols = one_hot_encoder.fit_transform(df[categorical_cols])
encoded_categorical_cols = pd.DataFrame(
    encoded_categorical_cols, 
    columns=one_hot_encoder.get_feature_names_out(categorical_cols)
)

# Concatenate numerical and categorical columns
X = pd.concat([df[numerical_cols], encoded_categorical_cols], axis=1)
y = df['Flood Risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Scale the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and compile the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

# Save the model and scaler
model.save('flood_model.keras') 
# Save feature names
joblib.dump(X.columns.tolist(), 'feature_names.joblib')
