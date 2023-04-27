import pandas as pd
import numpy as np
import streamlit as st
import joblib
import pickle
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.ensemble import RandomForestClassifier 



# with open('rta_model_deploy_c.joblib', 'rb') as f_in:
#     cctx = zstd.ZstdCompressor(level=10)
#     with open('my_compress.joblib.zst', 'wb') as f_out:
#         writer = cctx.stream_writer(f_out)
#         writer.write(f_in.read())
#         writer.flush(zstd.FLUSH_FRAME)
        
        
# with open('my_compress.joblib.zst', 'rb') as compressed_file:
#     # Create a decompression context
#     dctx = zstd.ZstdDecompressor()

#     # Create a decompression stream
#     with dctx.stream_reader(compressed_file) as reader:
#         # Open a new file for writing the decompressed data
#         with open('decompressed_new_file.joblib', 'wb') as decompressed_file:
#             # Decompress the data and write it to the output file
#             decompressed_file.write(reader.read())


# splitted_filenames = ["rta_model_deploy.joblib.partaa", "rta_model_deploy.joblib.partab", "rta_model_deploy.joblib.partac","rta_model_deploy.joblib.partad", "rta_model_deploy.joblib.partae", "rta_model_deploy.joblib.partaf","rta_model_deploy.joblib.partai", "rta_model_deploy.joblib.partaj", "rta_model_deploy.joblib.partak","rta_model_deploy.joblib.partal", "rta_model_deploy.joblib.partam"]


# with open('model.joblib', 'wb') as outfile:
#     for f in splitted_filenames:
#         with open(f, 'rb') as infile:
#             shutil.copyfileobj(infile, outfile)

# model = joblib.load("rta_model_deploy.joblib")
# encoder = joblib.load("ordinal_encoder.joblib")
# model=pickle.load('model_saved.pickle')
# encoder=pickle.load('ordinal_encoder.pickle')


# with open('model_savedrf_latest.pickle', 'rb') as f:
#     model = pickle.load(f)

with open('model_savedrf_latest.pickle', 'rb') as f:
     modellr = pickle.load(f)

with open('ordinal_encoderrf_latest.pickle', 'rb') as f:
    encoder = pickle.load(f)

st.set_option('deprecation.showPyplotGlobalUse', False)



st.set_page_config(page_title="Accident Severity Prediction",
                page_icon="🚧", layout="wide")

#creating option list for dropdown menu

options_city = [
      'Dayton','Antioch',
 'Arlington',
 'Atlanta',
 'Austin',
 'Bakersfield',
 'Barrington',
 'Baton Rouge',
 'Bloomingdale',
 'Bradenton',
 'Bronx',
 'Brooklyn',
 'Buffalo Grove',
 'Charlotte',
 'Chattanooga',
 'Chicago',
 'Colorado Springs',
 'Columbia',
 'Corbett',
 'Dallas',
 
 'Denton',
 'Denver',
 'Des Moines',
 'Detroit',
 'Downers Grove',
 'Elgin',
 'Florence',
 'Fort Lauderdale',
 'Fort Myers',
 'Frederick',
 'Fredericksburg',
 'Fresno',
 'Grand Rapids',
 'Grayslake',
 'Greensboro',
 'Greenville',
 'Gurnee',
 'Hialeah',
 'Homestead',
 'Houston',
 'Indianapolis',
 'Jacksonville',
 'Kansas City',
 'Kissimmee',
 'Lake Forest',
 'Lake Zurich',
 'Lancaster',
 'Lexington',
 'Libertyville',
 'Linden',
 'Los Angeles',
 'Los Gatos',
 'Madison',
 'Memphis',
 'Miami',
 'Minneapolis',
 'Monroe',
 'Mundelein',
 'Nashville',
 'New Castle',
 'New Orleans',
 'Newark',
 'North Miami',
 'Ogden',
 'Oklahoma City',
 'Orlando',
 'Pensacola',
 'Philadelphia',
 'Phoenix',
 'Pittsburgh',
 'Portland',
 'Pueblo',
 'Raleigh',
 'Richmond',
 'Rochester',
 'Round Lake',
 'Sacramento',
 'Saint Paul',
 'Saint Petersburg',
 'Salem',
 'Salt Lake City',
 'San Antonio',
 'San Bernardino',
 'San Diego',
 'San Francisco',
 'Sarasota',
 'Seattle',
 'Shreveport',
 'Silver Spring',
 'Springfield',
 'Stockton',
 'Syracuse',
 'Tallahassee',
 'Tampa',
 'Tucson',
 'Tulsa',
 'Washington',
 'Waukegan',
 'Waukesha',
 'Whittier',
 'Winston Salem',
 'Woodbridge',
 'York',
 'Zion']


options_state = ['OH', 'IN', 'KY', 'WV', 'MI', 'PA', 'CA', 'NV', 'MN', 'TX', 'MO', 'CO', 'OK', 'LA', 'KS', 'WI', 'IA', 'MS', 'NE', 'ND', 'WY', 'SD', 'MT', 'NM', 'AR', 'IL', 'NJ', 'GA', 'FL', 'NY', 'CT', 'RI', 'SC', 'NC', 'MD', 'MA', 'TN', 'VA', 'DE', 'DC', 'ME', 'AL', 'NH', 'VT', 'AZ', 'UT', 'ID', 'OR', 'WA']

options_wind_direction = ['CALM',        
'SouthWest',    
'SouthEast',    
'NorthEast',    
'North',         
'NorthWest',     
'South',         
'West',          
'East',          
'Variable']

#options_junction = [False, True]

options_traffic_signal = [0,1]

options_weather_condition = ['Clear',                    
'Cloudy',                     
'Scattered Clouds',            
'Rain',                       
'Snow',                       
'Windy',                      
'Thunderstorm',               
'Other',                        
'Light Rain with Thunder']

options_sun = ['Day','Night']


#options_twilight = ['Day','Night']



# features list
features = ['City', 'Weather_Condition', 'Sunrise_Sunset', 'Wind_Direction',
       'Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
       'Visibility(mi)', 'Wind_Speed(mph)', 'Traffic_Signal', 'Hour_of_Day',
       'Day_of_Week']

# take input 
st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction 🚧</h1>", unsafe_allow_html=True)
def main():
       with st.form("accident_severity_form"):
              st.subheader("Please enter the following parameters:")
              
              Distance_miles = st.slider("Length of Traffic Jam in miles:",0,100, value=0, format="%d")
              Temperature = st.slider("Temperature in Fahrenheit:",-27,190, value= -25, format="%d")
              Wind_chill = st.slider("Wind Chill in F:", -48, 190, value= -50, format="%d")
              Humditity = st.slider("Humidity percentage:",1, 100, value=0, format="%d")
              #Pressure = st.slider("Pressure:",16, 60, value=0, format="%d")
              Visibility = st.slider("Visibility in miles",0, 100, value=0, format="%d")
              Wind_speed = st.slider("Wind Speed in mph",0, 1087, value=0, format="%d")
             # Precipitation = st.slider("Precipitation in inches",0, 10, value=0, format="%d")
            
             # State = st.selectbox("State:", options=options_state)
                
       
              City = st.selectbox("City", options=options_city)
                
              
              Weather_Condition = st.selectbox("Weather Condition", options=options_weather_condition)
              Wind_Direction = st.selectbox("Wind direction:", options=options_wind_direction)
              # Junction = st.selectbox("Junction:", options=options_junction)
              Traffic_Signal = st.selectbox("Traffic Signal:", options=options_traffic_signal)
              Sunrise_Sunset = st.selectbox("Day/Night:", options=options_sun)
             # Civil_Twilight = st.selectbox("Twilight:", options=options_twilight)
              Hour_of_Day = st.slider("Hour_of_Day",0, 23, value=0, format="%d")
              Day_of_Week = st.slider("Day_of_Week",0, 6, value=0, format="%d")
            
    
              
              submit = st.form_submit_button("Predict")

# encode using ordinal encoder and predict
       if submit:
              input_array = np.array([City,
       Weather_Condition,Sunrise_Sunset,Wind_Direction], ndmin=2)
              print(input_array)
              
              #encoded_arr = list(encoder.transform(input_array).ravel())
              encoded_arr = list(encoder.transform(input_array).ravel())
              print(encoded_arr)
              
              num_arr = [Distance_miles,Temperature,Wind_chill,Humditity,Visibility,Wind_speed,Traffic_Signal,Hour_of_Day,Day_of_Week]
              pred_arr = [encoded_arr+num_arr]   
              test = pd.DataFrame(pred_arr ,columns = ['City', 'Weather_Condition',
       'Sunrise_Sunset', 'Wind_Direction', 'Distance(mi)', 'Temperature(F)',
       'Wind_Chill(F)', 'Humidity(%)', 'Visibility(mi)',
       'Wind_Speed(mph)','Traffic_Signal','Hour_of_Day','Day_of_Week'])      
          
              #prediction = model.predict(test)
              prediction = modellr.predict(test)
              
              if prediction == 2:
                     st.write(f"The severity prediction is Low impact")
                     st.image("low.jpeg", width=200, height=100)
              elif prediction == 3:
                     st.write(f"The severity prediction is Medium Impact")
                     st.image("medium.jpeg", width=200, height=100)
              else:
                     st.write(f"The severity prediciton is High Impact")
                     st.image("high.jpeg", width=200, height=100)
                  
#               st.subheader("Explainable AI (XAI) to understand predictions")  
#               shap.initjs()
#               shap_values = shap.TreeExplainer(model).shap_values(pred_arr)
#               st.write(f"For prediction {prediction}") 
#               shap.force_plot(shap.TreeExplainer(model).expected_value[0], shap_values[0],
#                               pred_arr, feature_names=features, matplotlib=True,show=False).savefig("pred_force_plot.jpg", bbox_inches='tight')
#               img = Image.open("pred_force_plot.jpg")
#               st.image(img, caption='Model explanation using shap')
              
              st.write("Developed By: Team 12")
              
              

# post the image of the accident

a,b,c = st.columns([0.2,0.6,0.2])
with b:
  st.image("Car_crash.jpeg", use_column_width=True)




#st.markdown("Please find GitHub repository link of project: [Click Here](https://github.com/avikumart/Road-Traffic-Severity-Classification-Project)")                  
                  
if __name__ == '__main__':
   main()
    
   
                
    
                     
              

       
       


