import streamlit as st
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Plagiarism Detection System",
    page_icon="📄",
    layout="centered"
)


# ---------------------------------------------------
# Custom Styling
# ---------------------------------------------------
st.markdown("""
    <style>

    .main {
        padding-top: 20px;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }

    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# Text Preprocessing Function
# ---------------------------------------------------
def preprocess(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'\W+', ' ', text)

    return text


# ---------------------------------------------------
# Similarity Calculation
# ---------------------------------------------------
def calculate_similarity(text1, text2):

    documents = [text1, text2]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    # Cosine Similarity
    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return similarity * 100


# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.markdown(
    '<div class="title">📄 Plagiarism Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload two text files to compare similarity using TF-IDF and Cosine Similarity</div>',
    unsafe_allow_html=True
)


# File Uploaders
file1 = st.file_uploader(
    "Upload First File",
    type=["txt"]
)

file2 = st.file_uploader(
    "Upload Second File",
    type=["txt"]
)


# ---------------------------------------------------
# Main Logic
# ---------------------------------------------------
if file1 and file2:

    # Read files
    text1 = file1.read().decode("utf-8")
    text2 = file2.read().decode("utf-8")

    # Preprocess
    clean_text1 = preprocess(text1)
    clean_text2 = preprocess(text2)

    # Calculate Similarity
    similarity_percentage = calculate_similarity(
        clean_text1,
        clean_text2
    )

    # ---------------------------------------------------
    # Result Display
    # ---------------------------------------------------
    st.markdown("## Similarity Result")

    # Progress Bar
    st.progress(int(similarity_percentage))

    # Color Logic
    if similarity_percentage > 70:
        st.error(
            f"High Plagiarism Detected: {similarity_percentage:.2f}%"
        )

    elif similarity_percentage > 40:
        st.warning(
            f"Moderate Similarity Found: {similarity_percentage:.2f}%"
        )

    else:
        st.success(
            f"Low Plagiarism Detected: {similarity_percentage:.2f}%"
        )


   