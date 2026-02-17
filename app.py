import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import os
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="ELIFLIX",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Red, Black & White Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Container */
    .stApp {
        background: #000000;
    }
    
    /* Header with Red Accent */
    .netflix-header {
        background: linear-gradient(135deg, #000000 0%, #1a0000 50%, #000000 100%);
        padding: 40px 30px;
        border-radius: 15px;
        margin-bottom: 50px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(255, 0, 0, 0.35);
        border: 3px solid #ff0000;
    }
    
    .netflix-logo {
        font-size: 4em;
        font-weight: 800;
        color: #ff0000;
        letter-spacing: -3px;
        margin-bottom: 15px;
        text-shadow: 0 0 20px rgba(255, 0, 0, 0.6);
    }
    
    .netflix-tagline {
        font-size: 1.3em;
        color: #ffffff;
        font-weight: 400;
        letter-spacing: 1px;
    }
    
    /* Search Section */
    .search-section {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        padding: 35px;
        border-radius: 12px;
        margin-bottom: 45px;
        border: 3px solid #ff0000;
        box-shadow: 0 15px 45px rgba(255, 0, 0, 0.25);
    }
    
    .search-title {
        font-size: 1.9em;
        color: #ff0000;
        margin-bottom: 25px;
        font-weight: 700;
        letter-spacing: 1px;
        text-shadow: 0 0 10px rgba(255, 0, 0, 0.3);
    }
    
    /* Movie Card Styling */
    .movie-card {
        position: relative;
        overflow: hidden;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(255, 0, 0, 0.2);
        transition: all 0.35s cubic-bezier(0.23, 1, 0.320, 1);
        cursor: pointer;
        background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
        border: 2px solid #ff0000;
    }
    
    .movie-card:hover {
        transform: translateY(-12px) scale(1.03);
        box-shadow: 0 20px 50px rgba(255, 0, 0, 0.5);
        border-color: #ff3333;
    }
    
    .movie-poster-placeholder {
        background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
        position: relative;
        overflow: hidden;
    }
    
    .movie-poster-placeholder::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.15) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Info Cards */
    .movie-info {
        background-color: #0a0a0a;
        padding: 18px;
        border-radius: 10px;
        margin-top: 15px;
        border-left: 4px solid #ff0000;
        transition: all 0.3s ease;
    }
    
    .movie-info:hover {
        border-left-color: #ff3333;
        box-shadow: 0 5px 15px rgba(255, 0, 0, 0.3);
    }
    
    .movie-info-title {
        color: #ff0000;
        font-weight: 700;
        margin-bottom: 12px;
        font-size: 1.05em;
        text-shadow: 0 0 5px rgba(255, 0, 0, 0.2);
    }
    
    .movie-info-text {
        color: #ffffff;
        font-size: 0.95em;
        line-height: 1.6;
        font-weight: 400;
    }
    
    /* Recommendations Header */
    .recommendations-header {
        color: #ff0000;
        font-size: 2.2em;
        font-weight: 800;
        margin: 50px 0 30px 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(255, 0, 0, 0.3);
    }
    
    /* Stat Boxes */
    .stat-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
        border-left: 5px solid #ff0000;
        padding: 28px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 8px 24px rgba(255, 0, 0, 0.2);
        transition: all 0.3s ease;
        border: 1px solid #ff3333;
    }
    
    .stat-box:hover {
        border-left-color: #ff3333;
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(255, 0, 0, 0.35);
    }
    
    .stat-number {
        font-size: 2.5em;
        color: #ff0000;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(255, 0, 0, 0.3);
    }
    
    .stat-label {
        color: #ffffff;
        font-size: 1em;
        margin-top: 10px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Buttons - Professional Red Style */
    .stButton > button {
        background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
        color: white;
        border: 2px solid #ff0000;
        padding: 14px 35px;
        border-radius: 8px;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.35s cubic-bezier(0.23, 1, 0.320, 1);
        width: 100%;
        font-size: 1.05em;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 24px rgba(255, 0, 0, 0.35);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(255, 0, 0, 0.6);
        background: linear-gradient(135deg, #ff3333 0%, #ff0000 100%);
        border-color: #ff3333;
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Section Divider */
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #ff0000, transparent);
        margin: 50px 0;
        box-shadow: 0 0 10px rgba(255, 0, 0, 0.3);
    }
    
    /* Input Fields - Professional Styling */
    input, select, textarea {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
        border: 2px solid #ff0000 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 1em !important;
        transition: all 0.3s ease !important;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4) !important;
    }
    
    input:focus, select:focus, textarea:focus {
        border-color: #ff3333 !important;
        box-shadow: 0 0 0 4px rgba(255, 0, 0, 0.25), inset 0 2px 8px rgba(0, 0, 0, 0.4) !important;
        outline: none !important;
    }
    
    /* Streamlit Select Dropdown */
    .stSelectbox > div > div {
        background-color: #0a0a0a !important;
    }
    
    [data-baseweb="select"] {
        --color-primary: #ff0000 !important;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div {
        background-color: #1a1a1a !important;
    }
    
    .stSlider [role="slider"] {
        background-color: #ff0000 !important;
    }
    
    /* Placeholder Styling */
    input::placeholder, textarea::placeholder {
        color: #808080 !important;
        font-weight: 400 !important;
    }
    
    /* Text Input Focus with Glow */
    input[type="text"]:focus, input[type="number"]:focus, textarea:focus {
        background-color: #000000 !important;
    }
    
    /* Select Dropdown Styling */
    select > option {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Footer */
    footer {
        background: #000000;
        border-top: 2px solid #ff0000;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ff0000, #cc0000);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ff3333, #ff0000);
    }
    
    </style>
    """, unsafe_allow_html=True)

# TMDB Image URL base
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

# Load data from pickle files
@st.cache_data
def load_data():
    """Load preprocessed data from pickle files"""
    try:
        with open('pkl_models/df_clean.pkl', 'rb') as f:
            df_clean = pickle.load(f)
        return df_clean
    except FileNotFoundError:
        st.error("❌ Pickle files not found. Please run the notebook to generate them.")
        st.stop()

@st.cache_data
def load_similarity_matrix():
    """Load precomputed similarity matrix from pickle"""
    try:
        with open('pkl_models/similarity_matrix.pkl', 'rb') as f:
            similarity_matrix = pickle.load(f)
        return similarity_matrix
    except FileNotFoundError:
        st.error("❌ Similarity matrix not found. Please run the notebook.")
        st.stop()

@st.cache_data  
def get_movie_list():
    """Load movie list from pickle"""
    try:
        with open('pkl_models/movie_list.pkl', 'rb') as f:
            movie_list = pickle.load(f)
        return movie_list
    except FileNotFoundError:
        st.error("❌ Movie list not found. Please run the notebook.")
        st.stop()

def display_movie_card(row, width=200):
    """Display a professional movie card with poster placeholder"""
    rating = row.get('vote_average', 0)
    title = row.get('title', 'Unknown')
    year = str(row.get('release_date', 'N/A'))[:4] if row.get('release_date') else 'N/A'
    genres = row.get('genres_str', 'N/A')
    
    # Professional gradient placeholder with shimmer effect
    return f"""
    <div class='movie-card'>
        <div class='movie-poster-placeholder'
             style='width: {width}px; height: 300px; border-radius: 12px;
                     display: flex; align-items: center; justify-content: center;
                     color: white; font-size: 2.5em; font-weight: 800;
                     text-align: center; padding: 15px; position: relative;'>
            🎬
        </div>
        <div class='movie-info'>
            <div class='movie-info-title' style='font-size: 1.1em; margin-bottom: 10px;'>
                {title[:28]}
            </div>
            <div class='movie-info-text'>
                <strong>⭐ Rating:</strong> {rating:.1f}/10<br>
                <strong>📅 Year:</strong> {year}<br>
                <strong style='color: #ff0051;'>Genres:</strong><br>
                <span style='font-size: 0.85em; display: block; margin-top: 5px;'>{genres[:45]}</span>
            </div>
        </div>
    </div>
    """

def get_recommendations(movie_title, df_clean, similarity_matrix, n_recommendations=10):
    """Get movie recommendations"""
    matches = df_clean[df_clean['title'].str.lower().str.contains(movie_title.lower(), na=False)]
    
    if matches.empty:
        return None, None
    
    movie_idx = matches.index[0]
    movie_title_exact = df_clean.loc[movie_idx, 'title']
    
    sim_scores = list(enumerate(similarity_matrix[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(idx, score) for idx, score in sim_scores if idx != movie_idx]
    
    top_indices = [idx for idx, score in sim_scores[:n_recommendations]]
    top_scores = [score for idx, score in sim_scores[:n_recommendations]]
    
    recommendations = df_clean.loc[top_indices].copy()
    recommendations['similarity_score'] = top_scores
    
    return movie_title_exact, recommendations

# Load data from pickle files
df_clean = load_data()
similarity_matrix = load_similarity_matrix()
all_movies = get_movie_list()

# Initialize session state
if 'search_input' not in st.session_state:
    st.session_state.search_input = ""
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

# Header
st.markdown("""
    <div class='netflix-header'>
        <div class='netflix-logo'>🎬 ELIFLIX</div>
        <div class='netflix-tagline'>Discover Movies You'll Love</div>
    </div>
    """, unsafe_allow_html=True)

# Search Section
st.markdown("""
    <div class='search-section'>
        <div class='search-title'>🔍 Find Your Next Favorite Movie</div>
    </div>
    """, unsafe_allow_html=True)

# Search input
col1, col2 = st.columns([4, 1])

with col1:
    selected_movie = st.selectbox(
        "Select a movie:",
        all_movies,
        label_visibility="collapsed",
        placeholder="Search for a movie...",
        key="movie_select"
    )

with col2:
    n_recs = st.slider(
        "Results:",
        min_value=5,
        max_value=20,
        value=12,
        label_visibility="collapsed"
    )

# Search button
if st.button("🎬 GET RECOMMENDATIONS", use_container_width=True):
    if selected_movie:
        movie_title, recommendations = get_recommendations(
            selected_movie,
            df_clean,
            similarity_matrix,
            n_recs
        )
        
        if movie_title:
            st.session_state.selected_movie = movie_title
            st.session_state.recommendations = recommendations

# Display selected movie
if st.session_state.selected_movie:
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Get original movie info
    orig_movie = df_clean[df_clean['title'] == st.session_state.selected_movie].iloc[0]
    
    st.markdown(f"<div class='recommendations-header'>You Selected: {st.session_state.selected_movie}</div>", 
                unsafe_allow_html=True)
    
    # Display original movie with details
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown(display_movie_card(orig_movie, width=180), unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='movie-info'>
            <div class='movie-info-title' style='font-size: 1.5em;'>{orig_movie['title']}</div>
            <div class='movie-info-text'>
                <strong>Rating:</strong> ⭐ {orig_movie['vote_average']:.1f}/10<br>
                <strong>Release Year:</strong> 📅 {orig_movie.get('release_date', 'N/A')[:4]}<br>
                <strong>Genres:</strong> 🎭 {orig_movie['genres_str']}<br>
                <strong>Overview:</strong><br>
                {orig_movie['overview'][:300]}...
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display recommendations
    if st.session_state.recommendations is not None:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='recommendations-header'>📺 Recommended For You</div>", 
                    unsafe_allow_html=True)
        
        recommendations = st.session_state.recommendations
        
        # Create grid of movie cards
        cols = st.columns(4)
        
        for idx, row in recommendations.iterrows():
            col_idx = idx % 4
            with cols[col_idx]:
                st.markdown(display_movie_card(row), unsafe_allow_html=True)
                
                # Match percentage
                st.markdown(f"""
                <div class='movie-info-text' style='text-align: center; margin-top: 5px;'>
                    <strong>Match:</strong> {row['similarity_score']:.1%}
                </div>
                """, unsafe_allow_html=True)

# Home screen if nothing selected
if not st.session_state.selected_movie:
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Stats
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-number'>{len(df_clean)}</div>
            <div class='stat-label'>Movies Available</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-number'>{df_clean['vote_average'].mean():.1f}</div>
            <div class='stat-label'>Avg Rating</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-number'>{int(df_clean['vote_average'].max())}</div>
            <div class='stat-label'>Highest Rated</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Top rated section
    st.markdown(f"<div class='recommendations-header'>⭐ Top Rated Movies</div>", 
                unsafe_allow_html=True)
    
    top_movies = df_clean.nlargest(8, 'vote_average')
    
    cols = st.columns(4)
    
    for idx, (_, row) in enumerate(top_movies.iterrows()):
        col_idx = idx % 4
        with cols[col_idx]:
            st.markdown(display_movie_card(row), unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; color: #666; padding: 40px 20px; border-top: 1px solid #333; margin-top: 40px;'>
        <p style='color: #E50914; font-weight: bold;'>🎬 ELIFLIX</p>
        <small>Experience the Future of Movie Discovery | Powered by AI & Machine Learning</small>
    </div>
    """, unsafe_allow_html=True)
