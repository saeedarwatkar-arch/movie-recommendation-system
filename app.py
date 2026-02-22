import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title=" Movie Recommender Systen ", layout="wide")

# ---------------- NETFLIX STYLE CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
.movie-card img {
    border-radius: 12px;
    transition: transform 0.3s ease;
}
.movie-card img:hover {
    transform: scale(1.08);
}
.movie-title {
    text-align: center;
    font-weight: bold;
    font-size: 15px;
    padding-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA (DEPLOY SAFE) ----------------
@st.cache_resource
def load_data():
    # load movies list
    movies = pickle.load(open('model/movies_list.pkl','rb'))

    # rebuild similarity matrix instead of loading big file
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()

    similarity = cosine_similarity(vectors)

    return movies, similarity

movies, similarity = load_data()

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

# ---------------- UI HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🎬 Netflix Style Movie Recommendation System</h1>", unsafe_allow_html=True)

movie_list = movies['title'].values

selected_movie = st.selectbox(
    "🍿 Choose a movie you like",
    movie_list
)

# ---------------- BUTTON ----------------
if st.button("🔥 Show Recommendations"):

    with st.spinner("Finding similar movies..."):
        names, posters = recommend(selected_movie)

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.markdown(f"""
            <div class="movie-card">
                <img src="{posters[i]}" width="100%">
                <div class="movie-title">{names[i]}</div>
            </div>
            """, unsafe_allow_html=True)