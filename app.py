import streamlit as st
import pandas as pd
import requests
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from streamlit_extras.add_vertical_space import add_vertical_space
import base64

# Load pre-trained model and scaler
model = tf.keras.models.load_model('flood_model.keras')
scaler = joblib.load('scaler.joblib')

# Load dataset and feature names
df = pd.read_csv('flood_risk_dataset_india.csv')
feature_names = joblib.load('feature_names.joblib')  # Ensure this file exists and contains feature names

# Load the second CSV file
df2 = pd.read_csv('IndianWeatherRepository.csv')

# Rename 'Flood Occurred' to 'Flood Risk' for consistency with the model
df = df.rename(columns={'Flood Occurred': 'Flood Risk'})

# Preprocess dataset to ensure all columns are available
categorical_cols = ['Land Cover', 'Soil Type', 'Infrastructure']
numerical_cols = [col for col in df.columns if col not in categorical_cols + ['Flood Risk']]

# One-hot encode categorical columns
one_hot_encoder = OneHotEncoder(sparse_output=False)
encoded_categorical_cols = one_hot_encoder.fit_transform(df[categorical_cols])
encoded_categorical_cols = pd.DataFrame(encoded_categorical_cols, columns=one_hot_encoder.get_feature_names_out(categorical_cols))

# Concatenate numerical and categorical columns
X = pd.concat([df[numerical_cols], encoded_categorical_cols], axis=1)
y = df['Flood Risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure that the scaler is consistent with the model
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler if it does not exist
joblib.dump(scaler, 'scaler.joblib')

# Check feature shape consistency
model_input_shape = model.input_shape[1]  # Should match the number of features
current_feature_shape = X_train_scaled.shape[1]

if model_input_shape != current_feature_shape:
    st.error(f"Feature shape mismatch: Model expects {model_input_shape} features, but got {current_feature_shape} features.")
else:
    # Compute the accuracy
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    # Streamlit app layout
    st.set_page_config(page_title='HydroView Analytics', layout='wide')

    # Add custom CSS for styling
    custom_css = """
    <style>
    .custom-title {
        text-align: center;
        color: black;
        font-size: 50px;
        font-family: 'Times New Roman', sans-serif;
        font-weight: bold;
        margin-top: 50px;
    }
    .custom-subtitle {
        text-align: center;
        color: #333;
        font-size: 24px;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Display styled title and subtitle
    st.markdown('<div class="custom-title">HydroView Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-subtitle">Real-Time Flood AI Analysis</div>', unsafe_allow_html=True)

    # Convert image to base64
    def img_to_base64(img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    # Set local image path
    local_img_path = 'WhatsApp Image 2024-08-17 at 20.45.45_65285be2.jpg'  # Update this with your local image path
    img_base64 = img_to_base64(local_img_path)

    # Embed CSS with base64 image
    css_code = f"""
    <style>
    .stApp {{
        background-image: url('data:image/jpeg;base64,{img_base64}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)

    # Sidebar for navigation
    with st.sidebar:
        st.title("📢 FloodAlert!!")
        st.subheader("Predict. Prepare. Protect.")
        st.write('''
                 📌Greetings and welcome to HydroWatch Insights!

                 Your go-to tool for flood predictions. This tool helps you be educated and ready for floods by using deep learning algorithms and real-time data to forecast the likelihood of flooding in your location.''')
        
        option = st.selectbox("Explore Views", ('Flood Statistics', 'Predict Flood Risk'))

        st.write(''' 
                 <u>**Features:**</u>
                - **Flood Risk Analysis:** Enter your location to get a thorough evaluation of your flood risk.
                - **Real-Time Data Integration:** Get access to current weather information, river levels, and other critical variables.
                - **Interactive Maps:** Use Google Maps integration to see places on a moving map including images.
                - **Historical Data:** To gain deeper understanding, compare the current situation with past flood data.
                - **Preparedness Tips:** Obtain tailored advice on how to get ready for future flooding.
                With HydroWatch Insights, you can protect your neighborhood and stay ahead of the weather.
                 
                 <u>**Floods: Dos and Don'ts:**</u>
                 - <u>**Dos-**</u>

                  1.**Stay Informed:** Monitor local weather updates and flood alerts.
                 
                  2.**Prepare an Emergency Kit:** Include essentials like food, water, and medical supplies.
                 
                  3.**Evacuate Early:** Follow evacuation orders promptly to ensure safety.
                 
                  4.**Protect Valuables:** Move important documents and valuables to higher ground.
                 
                  5.**Check Insurance:** Ensure flood insurance coverage for your property.
                 
                 - <u>**Don'ts-**</u>

                  1.**Avoid Driving Through Water:** Never drive through flooded roads or areas.
                 
                  2.**Don’t Ignore Warnings:** Follow evacuation and safety instructions from authorities.
                 
                  3.**Avoid Contact with Floodwater:** Stay away from contaminated water to prevent illness.
                 
                  4.**Don’t Re-enter Unsafe Areas:** Wait for official clearance before returning home.
                 
                  5.**Don’t Use Electrical Appliances:** Avoid using electrical devices if they’ve been exposed to water.''',unsafe_allow_html=True)
        
        add_vertical_space(3)
        st.markdown("📍For reliable flood updates and information, check out these websites:\n\n")
        st.markdown("""
                - [Wikipedia](https://en.m.wikipedia.org/wiki/Flood)
                - [NDMA](https://www.ndma.gov.in/)
                - [UPSDMA](https://upsdma.up.nic.in/)
                - [ASSAMSDMA](https://asdma.assam.gov.in/)

                """)
        

    # Define the fixed API key
    WEATHER_API_KEY = 'ebb86a3c2112c8c8986ec1596ff156d7'  # Replace with your actual API key
    GOOGLE_MAPS_API_KEY = 'AIzaSyCvcspQiHU3NUiTBWeL55P9Rj8owz2oN18'  # Replace with your Google Maps API key

    # Fetch real-time weather data
    def fetch_real_time_data(lat, lon, api_key=WEATHER_API_KEY):
        url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather = {
                'Latitude': lat,
                'Longitude': lon,
                'Rainfall (mm)': data.get('rain', {}).get('1h', 0),
                'Temperature (°C)': data['main']['temp'],
                'Humidity (%)': data['main']['humidity'],
                'River Discharge (m³/s)': 0,  # Add default value
                'Water Level (m)': 0,  # Add default value
                'Elevation (m)': 213,  # Example static value
                'Population Density': 5000,  # Example static value
                'Historical Floods': 0,  # Example static value
            }

            # Convert to DataFrame
            data_df = pd.DataFrame([weather])

            # Add missing one-hot encoded features with default values
            for col in feature_names:
                if col not in data_df.columns:
                    data_df[col] = 0

            # Ensure the columns are in the correct order
            data_df = data_df[feature_names]
            
            return data_df
        else:
            st.error(f"Failed to retrieve data: {response.status_code}")
            return None

    # Function to predict flood risk
    def predict_flood_risk(model, scaler, data, threshold=0.6):  # Increased threshold to 0.6
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)
        # Format the text with Markdown
        prediction_text = f"**Model Prediction:** {prediction[0][0]:.6f}"
        risk_text = f"**Predicted Flood Risk:** {'High' if prediction[0][0] > threshold else 'Low'}"
        st.markdown(f"**{prediction_text}**")
        st.markdown(f"**{risk_text}**")
        return 'High' if prediction[0][0] > threshold else 'Low'

    # Function to fetch images from a location using Google Places API
    def fetch_images_from_location(lat, lon, api_key=GOOGLE_MAPS_API_KEY, num_images=2, max_width=800):
        url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=1500&key={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            photos = []
            for place in data['results']:
                if 'photos' in place:
                    for photo in place['photos'][:num_images]:
                        photo_reference = photo['photo_reference']
                        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth={max_width}&photoreference={photo_reference}&key={api_key}"
                        photos.append(photo_url)
                    if len(photos) >= num_images:
                        break
            return photos[:num_images]
        else:
            st.error(f"Failed to retrieve images: {response.status_code}")
            return []

    # Display options
    if option == 'Flood Statistics':
        st.subheader('Flood Incidents in India (2015 - 2021)')
        st.write(df.head())

        st.subheader('Flood Incidents in India 2023 ')
        middle_rows = df2.iloc[188:193]
        st.write(middle_rows)

        # Add images to the Flood Statistics section
        img_path1 = 'Warning Sign.jpg'  # Update this with your local image path
        img_path2 = 'Flood India Map.jpg'  # Update this with your local image path

        st.subheader('Flood Risk Counts by Category')
        st.image(img_path1, caption='Warning Sign Flood', use_column_width=True, width=90)

        st.subheader('Flood Zone In India')
        st.image(img_path2, caption='Flood India Map', use_column_width=True, width=40)
        
        # Group by state and sum precipitation
        state_flood_data = df2.groupby('region')['precip_mm'].sum().reset_index()
        # Create the bar graph
        plt.figure(figsize=(12, 8))
        sns.barplot(x='region', y='precip_mm', data=state_flood_data, palette='viridis')
        plt.title('Total Precipitation (mm) by State')
        plt.xlabel('State')
        plt.ylabel('Total Precipitation (mm)')
        plt.xticks(rotation=90, ha='right') 
        plt.tight_layout()
        st.subheader('Statewise Flood Statistics :')
        st.pyplot(plt)

        # Group by state and calculate average temperature and humidity
        st.subheader('Statewise Temperature and Humidity Plot :')
        state_avg = df2.groupby('region').agg({
        'temperature_celsius': 'mean',
        'humidity': 'mean'
         }).reset_index()
        # Plot
        plt.figure(figsize=(12, 8))
        sns.lineplot(x='region', y='temperature_celsius', data=state_avg, marker='o', label='Temperature (°C)', color='blue')
        sns.lineplot(x='region', y='humidity', data=state_avg, marker='o', label='Humidity (%)', color='red')
        plt.xlabel('State')
        plt.ylabel('Value')
        plt.title('Average Temperature and Humidity by State :')
        plt.xticks(rotation=90)  # Rotate state names for better visibility
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

        # Pie chart for 'condition_text' column (top 7 conditions)
        st.subheader("Pie Chart Of Weather Conditions :")
        condition_counts = df2['condition_text'].value_counts().nlargest(7)
        plt.figure(figsize=(4, 4))
        plt.pie(condition_counts, labels=condition_counts.index, autopct='%1.1f%%',textprops={'fontsize': 6} ,startangle=140, colors=sns.color_palette("pastel"))
        st.pyplot(plt)

        # Heatmap
        numerical_df = df.select_dtypes(include=['float64', 'int64'])

        # Compute the correlation matrix
        corr = numerical_df.corr()

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
        plt.title('Correlation Heatmap of Flood Prediction Variables')
        plt.show()

    elif option == 'Predict Flood Risk':
        st.subheader('Predict Flood Risk for a Given Location')

        # Get latitude and longitude input
        lat = st.number_input("Enter Latitude:", -90.0, 90.0, 0.0)
        lon = st.number_input("Enter Longitude:", -180.0, 180.0, 0.0)

        if st.button("Predict Flood Risk"):
            data = fetch_real_time_data(lat, lon)
            if data is not None:
                risk = predict_flood_risk(model, scaler, data)
                st.markdown(f"**Real-Time Data:**")
                st.write(data)

                
                # Display map with flood-prone area
                st.subheader('Area on Map')
                map_url = f'https://www.google.com/maps/embed/v1/place?key={GOOGLE_MAPS_API_KEY}&q={lat},{lon}'
                components.html(f'<iframe src="{map_url}" width="100%" height="500" frameborder="0" style="border:0" allowfullscreen></iframe>', height=500)


                # Fetch and display images of the location
                st.subheader('Images of the Location')
                images = fetch_images_from_location(lat, lon)
                for image_url in images:
                    st.image(image_url, use_column_width=True)
