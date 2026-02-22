import pickle
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Netflix Style Movie Recommender",
    layout="wide"
)

# ---------------- CUSTOM NETFLIX CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.stApp {
    background-color: #0e1117;
}
h1 {
    text-align: center;
}
.movie-card img {
    border-radius: 10px;
    transition: transform 0.3s ease;
}
.movie-card img:hover {
    transform: scale(1.08);
}
.movie-title {
    text-align: center;
    font-weight: bold;
    font-size: 16px;
    padding-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
movies = pickle.load(open('model/movies_list.pkl','rb'))
similarity = pickle.load(open('model/similarity.pkl','rb'))

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    names = []
    posters = []

    for i in distances[1:6]:
        names.append(movies.iloc[i[0]].title)
        posters.append(movies.iloc[i[0]].poster)

    return names, posters

# ---------------- HEADER ----------------
st.markdown(
    "<h1>🎬 Netflix Style Movie Recommendation System</h1>",
    unsafe_allow_html=True
)

st.write("")

# ---------------- SELECT BOX ----------------
movie_list = movies['title'].values

selected_movie = st.selectbox(
    "🍿 Choose a movie you like",
    movie_list
)

# ---------------- BUTTON ----------------
if st.button("🔥 Show Recommendations"):

    with st.spinner("Finding similar movies..."):
        names, posters = recommend(selected_movie)

    st.write("")
    st.markdown("## ⭐ Recommended For You")

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.markdown(f"""
            <div class="movie-card">
                <img src="{posters[i]}" width="100%">
                <div class="movie-title">{names[i]}</div>
            </div>
            """, unsafe_allow_html=True)