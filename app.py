import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# LOAD TRAINED ML MODEL
# ==========================================

@st.cache_resource
def load_model():

    # Logistic Regression
    model = joblib.load("models/logistic_regression_model.pkl")
    vectorizer = joblib.load("models/logistic_regression_vectorizer.pkl")

    # Naive Bayes
    nb_model = joblib.load("models/Naive_Bayes_model.pkl")
    nb_vectorizer = joblib.load("models/Naive_Bayes_vectorizer.pkl")

    return model, vectorizer, nb_model, nb_vectorizer


model, vectorizer, nb_model, nb_vectorizer = load_model()


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ReviewGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0B1120;
    color: #F1F5F9;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111B2E;
    border-right: 1px solid #24344F;
}

/* Main headings */
h1, h2, h3 {
    color: #F8FAFC !important;
}

/* Paragraphs */
p, label {
    color: #CBD5E1;
}

/* Header */
.hero {
    background: linear-gradient(120deg, #142B50, #173B70);
    padding: 35px;
    border-radius: 18px;
    border: 1px solid #28518A;
    margin-bottom: 25px;
}

.hero-title {
    color: #FFFFFF;
    font-size: 38px;
    font-weight: 800;
}

.hero-text {
    color: #CBDDF7;
    font-size: 17px;
    margin-top: 10px;
}

/* Cards */
.custom-card {
    background-color: #142037;
    padding: 23px;
    border-radius: 15px;
    border: 1px solid #263B5B;
    min-height: 145px;
    margin-bottom: 15px;
}

.card-label {
    color: #93A9C8;
    font-size: 14px;
}

.card-value {
    color: #60A5FA;
    font-size: 25px;
    font-weight: bold;
    margin-top: 10px;
}

.card-description {
    color: #AAB9CF;
    font-size: 13px;
    margin-top: 8px;
}

/* Analysis panel */
.analysis-panel {
    background-color: #142037;
    border: 1px solid #263B5B;
    border-radius: 16px;
    padding: 25px;
    margin-top: 15px;
}

/* Buttons */
.stButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 12px 22px;
    font-weight: 600;
    width: 100%;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

/* Text area */
.stTextArea textarea {
    background-color: #111B2E;
    color: white;
    border: 1px solid #344968;
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #263B5B;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.markdown("# 🛡️ ReviewGuard AI")

    st.caption("INTELLIGENT REVIEW ANALYSIS")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["🏠 Home", "🔍 Analyse Review", "ℹ️ About Project"],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### Project Information")

    st.caption("Domain: Retail / E-commerce")
    st.caption("Technology: NLP + Machine Learning")
    st.caption("Learning: Supervised Learning")

    st.divider()

    st.markdown("🟢 Website Status: Online")
    st.caption("Academic Case Study Project")


# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">🛡️ ReviewGuard AI</div>
        <div class="hero-text">
            Detect potentially fake product reviews using
            Natural Language Processing and Machine Learning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Welcome to ReviewGuard AI")

    st.write(
        "ReviewGuard AI is an intelligent review analysis "
        "system designed to help identify potentially fake "
        "and genuine product reviews in e-commerce platforms."
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-label">DOMAIN</div>
            <div class="card-value">E-Commerce</div>
            <div class="card-description">
                Online shopping review analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-label">TECHNOLOGY</div>
            <div class="card-value">NLP + ML</div>
            <div class="card-description">
                Text processing and classification
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="custom-card">
            <div class="card-label">LEARNING TYPE</div>
            <div class="card-value">Supervised</div>
            <div class="card-description">
                Learning from labelled review data
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("How Does It Work?")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("### 01")
        st.markdown("**Enter a Review**")
        st.caption(
            "The user enters a product review "
            "into the analysis system."
        )

    with step2:
        st.markdown("### 02")
        st.markdown("**NLP Processing**")
        st.caption(
            "Text preprocessing and TF-IDF "
            "feature extraction."
        )

    with step3:
        st.markdown("### 03")
        st.markdown("**ML Prediction**")
        st.caption(
            "A trained classification model "
            "predicts the review category."
        )

    st.divider()

    st.info(
        "Both Logistic Regression and Naive Bayes models are successfully integrated. Select a model and analyse product reviews using ReviewGuard AI."
    )


# ==========================================
# REVIEW ANALYSIS PAGE
# ==========================================

elif page == "🔍 Analyse Review":

    st.title("🔍 Review Analysis")

    st.write(
        "Enter an online product review below to analyse its text."
    )

    st.markdown("""
    <div class="analysis-panel">
        <h3>Review Detection System</h3>
        <p>
        Our system uses Natural Language Processing (NLP) and supervised machine learning to analyse product reviews and predict whether they are computer-generated or original human-written.
        </p>
    </div>
    """, unsafe_allow_html=True)

    review = st.text_area(
        "Enter your product review:",
        placeholder=(
            "Example: The product quality is excellent. "
            "The battery lasts long and delivery was on time..."
        ),
        height=180,
        max_chars=5000
    )

    st.caption("Maximum input length: 5000 characters")
    selected_model = st.selectbox(
        "Choose Machine Learning Algorithm:",
        ["Logistic Regression", "Naive Bayes"]
    )
    
    if st.button("🔍 Analyse Review", use_container_width=True):

        if not review.strip():
            st.warning("Please enter a product review first.")

        else:
                        # Select the model
            if selected_model == "Logistic Regression":
                current_model = model
                current_vectorizer = vectorizer
            else:
                current_model = nb_model
                current_vectorizer = nb_vectorizer

            # Convert review into TF-IDF features
            review_tfidf = current_vectorizer.transform([review])

            # Predict review category
            prediction = current_model.predict(review_tfidf)[0]

            if prediction == "CG":
                result = "Computer-generated"
            elif prediction == "OR":
                result = "Original human-written"
            else:
                result = "Unknown category"

            # Calculate confidence
            confidence = None

            if hasattr(current_model, "predict_proba"):
                probabilities = current_model.predict_proba(review_tfidf)[0]
                confidence = float(max(probabilities)) * 100

            # Save results
            st.session_state["analysis_result"] = {
                "review": review,
                "prediction": result,
                "confidence": confidence,
                "model_name": selected_model
            }

    # Display analysis results
    # Display analysis results
    st.divider()
    st.subheader("Analysis Results")

    if "analysis_result" in st.session_state:

        result = st.session_state["analysis_result"]

        with st.container(border=True):

            st.markdown("### Analysis Complete")

            col1, col2 = st.columns(2)

            with col1:
                st.caption("Prediction")
                st.subheader(result["prediction"])

            with col2:
                if result["confidence"] is not None:
                    st.metric(
                        "Model Confidence",
                        f'{result["confidence"]:.2f}%'
                    )

            with st.expander("View Analysed Review"):
                st.write(result["review"])

            st.caption(f'Prediction generated using {result["model_name"]}.')

    else:
        st.info("Enter a review and click Analyse Review to see the prediction.")

# ==========================================
# ABOUT PROJECT PAGE
# ==========================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About ReviewGuard AI")

    st.write(
        "ReviewGuard AI is an academic machine learning "
        "case study focused on detecting potentially "
        "fake product reviews."
    )

    st.divider()

    st.subheader("Project Objectives")

    st.markdown("""
    - Identify potentially fake product reviews.
    - Apply Natural Language Processing techniques.
    - Use supervised machine learning algorithms.
    - Compare classification model performance.
    - Support better understanding of online review credibility.
    """)

    st.subheader("Machine Learning Techniques")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-value">NLP</div>
            <div class="card-description">
                Text cleaning, preprocessing and TF-IDF.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-value">Supervised ML</div>
            <div class="card-description">
                Naive Bayes and Logistic Regression.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

    st.subheader("Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-value">Logistic Regression</div>
            <div class="card-description">
                Machine Learning Algorithm used for review classification.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-value">TF-IDF</div>
            <div class="card-description">
                Converts review text into numerical features.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.metric("Reported Test Accuracy", "88.75%")

    st.caption(
        "Accuracy reported by the model training team."
    )

    st.write("Prediction Categories:")
    st.write("- Computer-generated")
    st.write("- Original human-written")
    
    st.divider()

    st.subheader("Model Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Logistic Regression Accuracy",
            "88.75%"
        )

    with col2:
        st.metric(
            "Naive Bayes Accuracy",
            "87.18%"
        )

    st.caption(
        "Accuracy values are based on the model training team's "
        "reported results and the supplied Naive Bayes confusion matrix."
    )

    st.write("Logistic Regression achieved the higher reported accuracy.")
    
    st.subheader("Accuracy Visualization")

    fig, ax = plt.subplots()

    models = ["Logistic Regression", "Naive Bayes"]
    accuracy = [88.75, 87.18]

    ax.bar(models, accuracy)

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model Accuracy Comparison")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Proposed Innovation")

    st.write(
        "A machine learning-based review classification system "
        "that analyses review text and predicts whether it is "
        "computer-generated or original human-written, along "
        "with a model confidence score."
    )

    st.divider()

    st.subheader("Project Domain")

    st.write("Retail / E-commerce")

    st.write("Subdomain: Natural Language Processing")

    st.divider()

    st.subheader("Project Limitations")

    st.markdown("""
    - Prediction accuracy depends on the quality of training data.
    - Some human-written reviews may be classified as computer-generated.
    - The model analyses review text and does not verify the actual product experience.
    """)

    st.subheader("Future Scope")

    st.markdown("""
    - Train the models using larger and more diverse datasets.
    - Explore advanced NLP and deep learning techniques.
    - Develop a more detailed explanation of prediction results.
    - Extend the system to support multiple languages.
    """)

    st.caption(
        "ReviewGuard AI | Academic Machine Learning Case Study"
    )
