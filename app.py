
import streamlit as st
import pandas as pd
import joblib
import re
import string
import os

st.set_page_config(
    page_title="College AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.hero {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(120deg, #142b52, #2874a6);
    color: white;
    text-align: center;
}
.result {
    padding: 20px;
    border-radius: 12px;
    background-color: #e8f4ff;
    border: 1px solid #b8d9f5;
}
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(
        "college_intent_model.pkl"
    )

@st.cache_data
def load_data():
    return pd.read_csv(
        "college_query_dataset.csv"
    )

def clean_text(text):
    text = text.lower()
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )
    text = re.sub(r'\s+', ' ', text).strip()
    return text

model = load_model()
df = load_data()

# Sidebar
st.sidebar.title("🎓 College AI")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "AI Query Assistant", "Dashboard", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "AI-based College Query Intent Classification System"
)

# HOME PAGE
if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>🎓 College AI Assistant</h1>
        <h3>Smart College Query Classification System</h3>
        <p>Ask your college-related questions
        and let AI identify their category.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.subheader("Welcome to College AI Assistant")

    st.write(
        "This system uses Natural Language Processing "
        "and Machine Learning to classify student queries."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Query Categories", "6")
    col2.metric("Dataset Questions", len(df))
    col3.metric("ML Model", "Logistic Regression")

    st.subheader("Available Categories")

    categories = [
        "Admission", "Fees", "Exam",
        "Hostel", "Scholarship", "Timetable"
    ]

    cols = st.columns(3)

    for i, category in enumerate(categories):
        with cols[i % 3]:
            st.info("📌 " + category)

    st.subheader("Example Questions")

    st.write("- How can I apply for admission?")
    st.write("- What is my semester fee?")
    st.write("- When will examinations start?")
    st.write("- Is hostel accommodation available?")

# AI QUERY ASSISTANT
elif page == "AI Query Assistant":

    st.title("🤖 AI Query Assistant")

    st.write(
        "Enter your college-related question below."
    )

    question = st.text_area(
        "Your Question",
        placeholder="Example: When will my semester exams start?",
        height=120
    )

    if st.button("Predict Query Category", type="primary"):

        if not question.strip():
            st.warning("Please enter a question first.")

        else:
            cleaned = clean_text(question)

            prediction = model.predict([cleaned])[0]

            probabilities = model.predict_proba([cleaned])[0]
            confidence = max(probabilities) * 100

            st.markdown("---")
            st.subheader("AI Prediction Result")

            st.success(
                f"Predicted Category: {prediction}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.caption(
                "Confidence is the model's predicted probability, "
                "not a guarantee that the classification is correct."
            )

            st.write("Category Probability:")

            probability_df = pd.DataFrame({
                "Category": model.classes_,
                "Probability": probabilities
            }).sort_values(
                "Probability", ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Category")
            )

# DASHBOARD
elif page == "Dashboard":

    st.title("📊 Project Dashboard")

    st.write(
        "Overview of the dataset used for model training."
    )

    col1, col2 = st.columns(2)

    col1.metric("Total Questions", len(df))
    col2.metric(
        "Total Categories",
        df["intent"].nunique()
    )

    st.subheader("Category Distribution")

    category_counts = df["intent"].value_counts()

    st.bar_chart(category_counts)

    st.subheader("Dataset Preview")

    st.dataframe(
        df[["text", "intent"]].head(20),
        use_container_width=True
    )

# ABOUT PROJECT
elif page == "About Project":

    st.title("📚 About the Project")

    st.write("""
    College Query Intent Classification System is an
    AI-based application that identifies the intent
    behind student questions.

    **Technologies Used:**

    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF Vectorization
    - Logistic Regression
    - Streamlit
    - DistilBERT (separate model experiment)

    **Project Categories:**

    1. Admission
    2. Fees
    3. Exam
    4. Hostel
    5. Scholarship
    6. Timetable

    **Developer:** Student Capstone Project
    """)

    st.info(
        "This is an intent classification system. "
        "It identifies query categories; it does not "
        "provide verified college-specific answers."
    )

st.markdown("---")
st.caption(
    "College AI Assistant | Machine Learning Capstone Project"
)
