import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import pickle

st.title('Course Recommender System')

@st.cache_data
def load_data_and_models():
    try:
        courses = pd.read_csv('model_data/courses.csv')
        user_interactions = pd.read_csv('model_data/user_interactions.csv')
        course_embeddings = np.load('model_data/course_embeddings.npy')
        with open('model_data/user_encoder.pkl', 'rb') as f:
            user_encoder = pickle.load(f)
        with open('model_data/course_encoder.pkl', 'rb') as f:
            course_encoder = pickle.load(f)
        ann_model = tf.keras.models.load_model('model_data/ann_model.h5')
        return courses, user_interactions, course_embeddings, user_encoder, course_encoder, ann_model
    except Exception as e:
        st.error(f"Error loading data or models: {e}")
        return None, None, None, None, None, None

courses, user_interactions, course_embeddings, user_encoder, course_encoder, ann_model = load_data_and_models()
if courses is None:
    st.stop()
else:
    st.write("Models and data loaded successfully.")
    st.write(f"Number of Courses Available: {len(courses)}")
    st.write(f"Number of Users in Data: {user_interactions['user_id'].nunique()}")
    st.write(f"User ID Range: {user_interactions['user_id'].min()} to {user_interactions['user_id'].max()}")

st.sidebar.header('User Preferences')
user_id = st.sidebar.number_input('Enter User ID', min_value=int(user_interactions['user_id'].min()), max_value=int(user_interactions['user_id'].max()), value=int(user_interactions['user_id'].min()))
num_recommendations = st.sidebar.slider('Number of Recommendations', 1, 10, 5)

def get_content_recommendations(num_recs):
    all_courses = courses
    if all_courses.empty:
        return pd.DataFrame()
    course_indices = all_courses.index
    if len(course_indices) > 1:
        similarity_matrix = cosine_similarity(course_embeddings[course_indices])
        sim_scores = list(enumerate(similarity_matrix[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recs+1]
        selected_indices = [i[0] for i in sim_scores]
        return all_courses.iloc[selected_indices]
    else:
        return all_courses.head(num_recs)

def get_collaborative_recommendations(user_id, num_recs):
    if user_id not in user_interactions['user_id'].unique():
        st.write(f"Debug: User ID {user_id} not found in interaction data.")
        return pd.DataFrame()
    
    user_encoded = user_encoder.transform([user_id])[0] if user_id in user_encoder.classes_ else -1
    if user_encoded == -1:
        st.write(f"Debug: User ID {user_id} could not be encoded.")
        return pd.DataFrame()
    
    taken_courses = user_interactions[user_interactions['user_id'] == user_id]['course_id'].tolist()
    available_courses = courses[~courses['course_id'].isin(taken_courses)]
    if available_courses.empty:
        st.write(f"Debug: No available courses for User ID {user_id} (all courses already taken).")
        return pd.DataFrame()
    
    available_course_ids = available_courses['course_id'].values
    try:
        available_course_encoded = course_encoder.transform(available_course_ids)
    except ValueError as e:
        seen_course_ids = [cid for cid in available_course_ids if cid in course_encoder.classes_]
        if not seen_course_ids:
            st.write(f"Debug: No known course IDs available for prediction after filtering.")
            return pd.DataFrame()
        available_courses = available_courses[available_courses['course_id'].isin(seen_course_ids)]
        available_course_ids = available_courses['course_id'].values
        available_course_encoded = course_encoder.transform(available_course_ids)
    
    user_input = np.full_like(available_course_encoded, user_encoded)
    
    try:
        predicted_ratings = ann_model.predict([user_input, available_course_encoded], verbose=0)
    except Exception as e:
        st.write(f"Debug: Error in prediction for User ID {user_id}: {e}")
        return pd.DataFrame()
    
    predictions = pd.DataFrame({
        'course_id': available_course_ids,
        'predicted_rating': predicted_ratings.flatten()
    })
    
    top_recs = predictions.sort_values(by='predicted_rating', ascending=False).head(num_recs)
    return courses[courses['course_id'].isin(top_recs['course_id'])]

st.header('Your Course Recommendations')
if st.button('Get Recommendations'):
    with st.spinner('Generating recommendations...'):
        content_recs = get_content_recommendations(num_recommendations)
        collab_recs = get_collaborative_recommendations(user_id, num_recommendations)

        st.subheader('Based on Content Similarity')
        if not content_recs.empty:
            st.dataframe(content_recs[['course_id', 'title']])
        else:
            st.write('No content-based recommendations available.')

        st.subheader('Based on User Interactions (ANN Model)')
        if not collab_recs.empty:
            st.dataframe(collab_recs[['course_id', 'title']])
        else:
            st.write('No collaborative recommendations available for this user.')
else:
    st.write('Click the button to generate recommendations based on your preferences.')

st.markdown('---')
st.markdown('Built with Streamlit for personalized course recommendations.')