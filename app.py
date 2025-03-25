import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('modified_movies.csv')

cv = CountVectorizer(max_features=10000, stop_words='english')

vec = cv.fit_transform(data['tags'].values.astype('U')).toarray()

s = cosine_similarity(vec)

dist = sorted(list(enumerate(s[0])), reverse=True, key=lambda  vec: vec[1])

def recommed_movie(mov):
    id = data[data['original_title'] == mov].index[0]
    distance = sorted(list(enumerate(s[id])), reverse=True, key=lambda  vec: vec[1])
    mov_list = []
    for i in distance[:5]:
        mov_list.append(data.iloc[i[0]].original_title)
    return mov_list

st.image('C:\\Users\\user\\Desktop\\movie\\msba.png')

st.title('Movie Recommendation System')

st.header('MSBA 315 - Group 7')

with st.container():
    st.markdown("""
    <div style="background-color:#1f1f1f;padding:20px;border-radius:10px">
        <h3 style="color:white;">Participants:</h3>
        <h4 style="color:white;">• Motorcycle Dude</h4>
        <h4 style="color:white;">• Cat Woman</h4>
        <h4 style="color:white;">• Alex's Mom</h4>
        <h4 style="color:white;">• Sergi Lavrov</h4>
        <h4 style="color:white;">• Sleepless Man</h4>
    </div>
    """, unsafe_allow_html=True)

movie =  st.selectbox('Select the Movie:', data['original_title'])

butt = st.button('Recommend')

if butt:
    pred_movies = recommed_movie(movie)
    st.subheader(f'The recommended movies similar to {movie} are:')
    for i in pred_movies:
        st.write(i)
