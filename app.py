import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Student Math Score Prediction')

def user_input_features():
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gender',('male','female'))
        race_ethnicity = st.selectbox('Race/Ethnicity',('group A','group B', 'group C', 'group D', 'group E'))
        parental_level_of_education = st.selectbox("Parent's Education Level",("bachelor's degree", 'some college', "master's degree", "associate's degree", 'high school', 'some high school'))
        reading_score = st.number_input('Reading Score', min_value=0, max_value=100, step=1)
        
    with col2:
        lunch = st.selectbox('Lunch',('standard', 'free/reduced'))
        test_preparation_course = st.selectbox('Test preparation course',('none', 'completed'))
        writing_score = st.number_input('Writing Score', min_value=0, max_value=100, step=1)
    data = {'gender': gender,
            'race_ethnicity': race_ethnicity,
            'parental_level_of_education': parental_level_of_education,
            'lunch': lunch,
            'test_preparation_course': test_preparation_course,
            'reading_score': reading_score,
            'writing_score': writing_score}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

if st.button('Predict'):
    # Combines user input features with entire student performance dataset
    # This will be useful for the encoding phase
    studs_raw = pd.read_csv('StudentsPerformance.csv')
    studs_raw.rename(columns = {'race/ethnicity' : 'race_ethnicity', 'parental level of education' : 'parental_level_of_education', 'test preparation course':'test_preparation_course', 'math score':'math_score', 'reading score':'reading_score', 'writing score':'writing_score'}, inplace=True)
    studs = studs_raw.drop(columns=['math_score'])
    df = pd.concat([input_df, studs],axis=0)
    
    # Encoding of numerical and categorical features
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    
    num_features = ['writing_score','reading_score']
    cat_features = ['race_ethnicity', 'parental_level_of_education', 'test_preparation_course', 'gender', 'lunch']
    
    oh_transformer = OneHotEncoder()
    num_transformer = StandardScaler()
    
    input_preprocessor = ColumnTransformer([
    ("OneHotEncoder", oh_transformer, cat_features),
    ("StandardScaler", num_transformer, num_features)
    ])
    
    df = input_preprocessor.fit_transform(df)
    df = df[:1] # Selects only the first row (the user input data)
    # Displays the user input features
    st.subheader('User Input features')
    st.write(input_df)
    
    # Reads in saved classification model
    fin_model = pickle.load(open('model.pkl', 'rb'))
    
    # Apply model to make predictions
    prediction = fin_model.predict(df)

    st.subheader('Prediction')
    st.write('The predicted maths score is :', prediction)
    
    