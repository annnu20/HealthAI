"""
AI-Powered Healthcare Diagnosis Assistant — Streamlit frontend.

Wraps the existing notebook backend (see backend/model.py,
backend/knowledge_base.py, backend/chatbot.py) with a multipage
Streamlit UI: Home, Diagnose, Chatbot, History, About, Account.

Patient accounts and prediction history are backed by a local SQLite
database (backend/database.py) — no external service required.

Run with:
    python -m streamlit run app.py
"""

import streamlit as st
import json
import os
import pandas as pd

from backend.chatbot import (
    analyze_report_text,
    answer_question,
    extract_text_from_pdf,
    analyze_medical_image,
)
from backend.database import (
    UserAlreadyExistsError,
    ValidationError,
    authenticate_user,
    clear_prediction_history,
    get_prediction_history,
    init_db,
    register_user,
    save_prediction,
)
from backend.knowledge_base import disease_details
from backend.model import ModelNotTrainedError, get_predictor

st.set_page_config(
    page_title="HealthAI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global theme — a calm indigo/teal healthcare palette, a rounded card
# language reused across every page, and a styled hero + feature grid for
# the Home page specifically.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
        h1, h2, h3, .hero-title, .feature-title { font-family: 'Poppins', sans-serif; }

        :root {
            --brand-indigo: #4F46E5;
            --brand-teal: #14B8A6;
            --brand-ink: #1F2937;
            --brand-muted: #6B7280;
        }

        .main > div { padding-top: 1.2rem; }

        /* Buttons */
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            border-radius: 10px;
            border: none;
            font-weight: 600;
            transition: transform 0.05s ease-in-out;
        }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--brand-indigo), var(--brand-teal));
            color: white;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); }

        /* Sidebar — carry the brand gradient through here too, and turn
           the plain radio-button nav into a pill-style menu instead of
           default circles on a mismatched light-gray background. */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--brand-indigo) 0%, #3730A3 100%);
            border-right: none;
        }
        section[data-testid="stSidebar"] * { color: #F9FAFB !important; }
        section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.25); }

        /* Nav "tabs" — rendered as st.button, one per page, styled as a
           stacked tab list instead of Streamlit's default radio circles */
        section[data-testid="stSidebar"] .stButton > button {
            justify-content: flex-start;
            text-align: left;
            background: transparent;
            border: 1px solid transparent;
            border-radius: 10px;
            padding: 0.55rem 0.9rem;
            margin-bottom: 0.2rem;
            color: white !important;
            font-weight: 500;
            box-shadow: none;
        }
        section[data-testid="stSidebar"] .stButton > button p { text-align: left; }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: transparent;
        }
        /* Active tab: left accent bar + brighter fill, instead of the
           gradient used for primary buttons elsewhere — reads more like a
           "selected tab" than a call-to-action button */
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: rgba(255, 255, 255, 0.22) !important;
            border-left: 4px solid var(--brand-teal) !important;
            font-weight: 700 !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            background: rgba(255, 255, 255, 0.28) !important;
        }

        /* Sidebar status caption (🟢 System ready) */
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            opacity: 0.9;
        }

        /* Hero */
        .hero-container {
            background: linear-gradient(135deg, var(--brand-indigo) 0%, var(--brand-teal) 100%);
            border-radius: 22px;
            padding: 3rem 2.5rem;
            text-align: center;
            color: white;
            box-shadow: 0 12px 30px rgba(79, 70, 229, 0.22);
            margin-bottom: 2rem;
        }
        .hero-title { font-size: 2.4rem; font-weight: 700; margin-bottom: 0.5rem; }
        .hero-subtitle {
            font-size: 1.08rem; opacity: 0.95; max-width: 640px;
            margin: 0 auto; line-height: 1.5;
        }
        .hero-badge {
            display: inline-block; background: rgba(255,255,255,0.18);
            padding: 0.3rem 0.9rem; border-radius: 999px; font-size: 0.85rem;
            margin-bottom: 1rem; font-weight: 500;
        }

        /* Feature cards */
        .feature-card {
            background: white; border-radius: 16px; padding: 1.4rem 1.3rem;
            border: 1px solid #EEF0F5; box-shadow: 0 4px 14px rgba(17, 24, 39, 0.04);
            height: 100%;
        }
        .feature-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
        .feature-title { font-weight: 600; font-size: 1.02rem; color: var(--brand-ink); margin-bottom: 0.35rem; }
        .feature-text { color: var(--brand-muted); font-size: 0.9rem; line-height: 1.45; }

        /* Step badges for "How it works" */
        .step-card { text-align: center; padding: 0.5rem 0.75rem; }
        .step-badge {
            display: inline-flex; align-items: center; justify-content: center;
            width: 40px; height: 40px; border-radius: 50%;
            background: linear-gradient(135deg, var(--brand-indigo), var(--brand-teal));
            color: white; font-weight: 700; font-family: 'Poppins', sans-serif;
            margin-bottom: 0.6rem;
        }
        .step-title { font-weight: 600; color: var(--brand-ink); margin-bottom: 0.25rem; }
        .step-text { color: var(--brand-muted); font-size: 0.87rem; }

        .disclaimer-box {
            background-color: #fff3cd;
            border: 1px solid #ffe69c;
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            font-size: 0.92rem;
            color: #664d03;
            margin-top: 1rem;
        }
        .upload-ack-box {
            background-color: #EEF2FF;
            border: 1px solid #C7D2FE;
            border-radius: 10px;
            padding: 0.7rem 1rem;
            font-size: 0.88rem;
            color: #3730A3;
            margin: 0.4rem 0 1rem 0;
        }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Database init (creates tables on first run; no-op afterwards)
# ---------------------------------------------------------------------------
init_db()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # each: {"role", "content", "image": bytes|None}

if "chat_uploader_key" not in st.session_state:
    st.session_state.chat_uploader_key = 0

if "user" not in st.session_state:
    st.session_state.user = None  # {"id": ..., "username": ..., "email": ...}

PAGES = ["Home", "Diagnose", "Chatbot", "History", "About", "Account"]
PAGE_LABELS = ["🏠 Home", "🚀 Diagnose", "💬 Chatbot", "📊 History", "ℹ️ About", "🔐 Account"]

if "page" not in st.session_state:
    st.session_state.page = "Home"


# ---------------------------------------------------------------------------
# Load the model once (cached across reruns).
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_predictor():
    return get_predictor()


model_load_error = None
predictor = None
try:
    predictor = load_predictor()
except ModelNotTrainedError as e:
    model_load_error = str(e)


# ---------------------------------------------------------------------------
# Sidebar navigation — rendered as a stacked "tab" list of buttons rather
# than a native radio group, so the active page can be styled as a
# highlighted tab (left accent bar) instead of a radio circle.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 HealthAI ")

    if st.session_state.user:
        st.markdown(f"**Signed in as** {st.session_state.user['username']}")
        if st.button("Log out", use_container_width=True):
            st.session_state.user = None
            st.session_state.chat_messages = []
            st.session_state.page = "Home"
            st.rerun()
        st.markdown("---")

    for name, label in zip(PAGES, PAGE_LABELS):
        is_active = st.session_state.page == name
        clicked = st.button(
            label,
            key=f"nav_{name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        )
        if clicked and not is_active:
            st.session_state.page = name
            st.rerun()

    #st.markdown("---")
    #if predictor is not None:
     #   st.caption("🟢 System ready")
    #else:
     #   st.caption("🔴 Model not loaded")

page = st.session_state.page


# ---------------------------------------------------------------------------
# Shared guards
# ---------------------------------------------------------------------------
def require_model():
    if predictor is None:
        st.error(
            "The trained model could not be loaded.\n\n"
            f"**Details:** {model_load_error}"
        )
        st.info(
            "From the project root, run:\n\n"
            "```bash\npython train_model.py\n```\n\n"
            "This trains the model on `data/Training.csv` and saves the "
            "artifacts into `models/`, then rerun the app."
        )
        st.stop()


def require_login():
    if st.session_state.user is None:
        st.warning("Please log in or create an account to use this page.")
        if st.button("Go to Account page"):
            st.session_state.page = "Account"
            st.rerun()
        st.stop()


# ---------------------------------------------------------------------------
# Page: Home — a clean, non-technical landing page. No model stats here;
# those live on the About page for anyone who wants them.
# ---------------------------------------------------------------------------
if page == "Home":
    greeting = f"Welcome back, {st.session_state.user['username']} 👋" if st.session_state.user else "Your health, understood a little better"

    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-badge">🩺 HealthAI </div>
            <div class="hero-title">{greeting}</div>
            <div class="hero-subtitle">
                Describe how you're feeling and get a clear, informative starting
                point — plus a chatbot you can talk to and even share your
                reports with. Simple, private, and available whenever you need it.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta_col1, cta_col2, cta_col3 = st.columns([1, 1, 1])
    with cta_col2:
        cta_label = "🚀 Check My Symptoms" if st.session_state.user else "🔐 Log In / Register to Get Started"
        if st.button(cta_label, type="primary", use_container_width=True):
            st.session_state.page = "Diagnose" if st.session_state.user else "Account"
            st.rerun()

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    feature_cols = st.columns(4)
    features = [
        ("🩺", "Symptom Checker", "Pick what you're feeling from a guided list and get an instant, informative prediction."),
        ("💬", "Talk to the Assistant", "Ask everyday health questions, or upload a report and get it acknowledged."),
        ("📁", "Private History", "Every check-in is saved securely to your account, ready whenever you look back."),
        ("🔒", "Your Account, Your Data", "Simple sign-up, securely hashed passwords, and nothing shared with anyone."),
    ]
    for col, (icon, title, text) in zip(feature_cols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

    st.markdown("#### How it works")
    step_cols = st.columns(3)
    steps = [
        ("1", "Create your account", "Takes less than a minute — just a username, email, and password."),
        ("2", "Tell us how you feel", "Select your symptoms, or chat naturally with the assistant."),
        ("3", "Get clear guidance", "See a likely condition, general advice, and save it for later."),
    ]
    for col, (num, title, text) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-badge">{num}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="disclaimer-box">⚠️ <b>Disclaimer:</b> This tool provides '
        "AI-generated information for general awareness only. It is not a "
        "substitute for professional medical diagnosis. Always consult a "
        "qualified healthcare provider.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Diagnose
# ---------------------------------------------------------------------------
elif page == "Diagnose":
    require_login()
    require_model()

    st.title("🚀 Diagnose Symptoms")
    st.markdown("Select the symptoms you're experiencing, then run the prediction.")

    tab_select, tab_text = st.tabs(["📋 Select from list", "✍️ Type symptoms"])

    selected_symptoms = []

    with tab_select:
        readable = {s: s.replace("_", " ").capitalize() for s in predictor.symptoms}
        display_to_raw = {v: k for k, v in readable.items()}
        chosen_display = st.multiselect(
            "Search and select symptoms",
            options=sorted(readable.values()),
            help="Start typing to filter the list of symptoms the model recognizes.",
        )
        selected_symptoms = [display_to_raw[d] for d in chosen_display]

    with tab_text:
        text_input = st.text_area(
            "Enter symptoms separated by commas",
            placeholder="itching, skin_rash, nodal_skin_eruptions",
            help="Use underscore_separated symptom names, comma-separated.",
        )
        if text_input.strip():
            selected_symptoms = [s.strip().lower() for s in text_input.split(",") if s.strip()]

    st.markdown("")
    run = st.button("🔍 Predict Disease", type="primary")

    if run:
        if not selected_symptoms:
            st.warning("Please select or enter at least one symptom before predicting.")
        else:
            with st.spinner("Analyzing symptoms..."):
                try:
                    disease, confidence, unrecognized = predictor.predict_with_confidence(
                        selected_symptoms
                    )
                    top3, _ = predictor.top_predictions(selected_symptoms, top_n=3)
                except Exception as e:
                    st.error("Something went wrong while running the prediction.")
                    st.caption(f"Technical detail: {e}")
                    st.stop()

            if unrecognized:
                st.info(
                    "Some entered symptoms weren't recognized and were ignored: "
                    + ", ".join(unrecognized)
                )

            st.success("Prediction complete")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(f"Predicted condition: **{disease.strip()}**")
            with col2:
                if confidence is not None:
                    st.metric("Confidence", f"{confidence:.1f}%")

            # If confidence is low, this is almost always because too few
            # symptoms were given to distinguish between diseases that
            # share them — not a modeling defect. Show what else is
            # commonly reported for the predicted condition instead of
            # hiding or inflating the number.
            if confidence is not None and confidence < 60:
                suggestions = predictor.common_symptoms_for(
                    disease, exclude=selected_symptoms
                )
                if suggestions:
                    readable_suggestions = [s.replace("_", " ") for s in suggestions]
                    st.info(
                        "Confidence is low because few symptoms were provided — many "
                        "conditions share a single symptom. People with this condition "
                        "commonly also report: **" + ", ".join(readable_suggestions) + "**. "
                        "Add any that apply and re-run for a more confident prediction."
                    )

            if len(top3) > 1:
                with st.expander("See top 3 likely conditions"):
                    for name, prob in top3:
                        if prob is not None:
                            st.write(f"**{name.strip()}** — {prob:.1f}%")
                            st.progress(min(int(prob), 100))
                        else:
                            st.write(f"**{name.strip()}**")

            details = disease_details(disease)
            st.markdown("### Health report")
            if details["available"]:
                st.markdown(f"**Description:** {details['description']}")
                st.markdown(f"**Recommended diet:** {details['diet']}")
                st.markdown("**Precautions:**")
                for p in details["precautions"]:
                    st.markdown(f"- ✔ {p}")
            else:
                st.warning(
                    "Detailed information for this condition isn't available in "
                    "the knowledge base yet."
                )

            st.markdown(
                '<div class="disclaimer-box">⚠️ This prediction is generated by an AI model. '
                "Always consult a qualified medical professional for diagnosis and treatment."
                "</div>",
                unsafe_allow_html=True,
            )

            save_prediction(
                user_id=st.session_state.user["id"],
                symptoms=", ".join(selected_symptoms),
                disease=disease.strip(),
                confidence=round(confidence, 1) if confidence is not None else None,
            )
            st.caption("Saved to your prediction history.")


# ---------------------------------------------------------------------------
# Page: Chatbot
# ---------------------------------------------------------------------------
elif page == "Chatbot":
    st.title("💬 HealthAI Chatbot")
    st.caption(
        "Ask about general topics — fever, diabetes, diet, sleep, stress, and "
        "more — or just say hello."
    )

    with st.expander("📎 Share a report, X-ray, or MRI (optional)"):
            st.caption(
                "Upload an X-ray, MRI image, or PDF report for AI-assisted analysis. "
                "The image analysis is educational only and is not a confirmed "
                "medical diagnosis or prescription."
            )
            uploaded_file = st.file_uploader(
                "Upload a PDF report, X-ray, or MRI image",
                type=["pdf", "png", "jpg", "jpeg"],
                key=f"chat_uploader_{st.session_state.chat_uploader_key}",
            )
            analyze_clicked = st.button(
                "🔍 Analyze & Send to Chat",
                disabled=uploaded_file is None
            )
    
            if analyze_clicked and uploaded_file is not None:
                file_bytes = uploaded_file.read()
                filename = uploaded_file.name
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    
                is_image = ext in ("png", "jpg", "jpeg")
    
                with st.spinner("🔍 Analyzing your upload..."):
                    if ext == "pdf":
                        extracted_text = extract_text_from_pdf(file_bytes)
                        answer = analyze_report_text(extracted_text, filename)

                    elif is_image:
                        mime_type = uploaded_file.type

                        answer = analyze_medical_image(
                            file_bytes=file_bytes,
                            mime_type=mime_type,
                            filename=filename,
                        )

                    else:
                        answer = (
                            f"I can't process **.{ext}** files yet — please upload a "
                            "PDF report, or a PNG/JPG image of a scan."
                        )
    
                st.session_state.chat_messages.append(
                    {
                        "role": "user",
                        "content": f"📎 Uploaded: {filename}",
                        "image": file_bytes if is_image else None,
                    }
                )
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": answer, "image": None}
                )
                # Reset the uploader widget for the next upload
                st.session_state.chat_uploader_key += 1
                st.rerun()
     
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg.get("image"):
                st.image(msg["image"], width=240)
            st.write(msg["content"])

    question = st.chat_input("Ask a health question, or just say hi...")
    if question:
        st.session_state.chat_messages.append(
            {"role": "user", "content": question, "image": None}
        )
        with st.chat_message("user"):
            st.write(question)

        answer = answer_question(question)
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer, "image": None}
        )
        with st.chat_message("assistant"):
            st.write(answer)

    if st.session_state.chat_messages:
        st.markdown("---")
        col1, col2 = st.columns(2)

        def _build_transcript():
            lines = []
            for msg in st.session_state.chat_messages:
                speaker = "You" if msg["role"] == "user" else "Assistant"
                lines.append(f"{speaker}: {msg['content']}")
            return "\n\n".join(lines)

        with col1:
            st.download_button(
                "⬇️ Download this chat",
                data=_build_transcript().encode("utf-8"),
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col2:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()


# ---------------------------------------------------------------------------
# Page: History
# ---------------------------------------------------------------------------
elif page == "History":
    require_login()

    st.title("📊 Prediction History")

    history = get_prediction_history(st.session_state.user["id"])

    if not history:
        st.info("No predictions yet. Head to the Diagnose page to run one.")
    else:
        history_df = pd.DataFrame(history)
        history_df = history_df.rename(
            columns={
                "symptoms": "Symptoms",
                "predicted_disease": "Predicted Disease",
                "confidence": "Confidence (%)",
                "created_at": "Date (UTC)",
            }
        )
        st.dataframe(history_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download history as CSV",
                data=history_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
            )
        with col2:
            if st.button("🗑️ Clear history"):
                clear_prediction_history(st.session_state.user["id"])
                st.rerun()


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
elif page == "About":
    st.title("ℹ️ About this project")

    st.markdown(
        """
**AI-Powered Healthcare Diagnosis Assistant** predicts likely diseases from
a set of reported symptoms, using a model trained on a labeled dataset of
132 symptoms across 41 diseases. Patient accounts and prediction history
are stored locally in SQLite.
        """
    )

    if predictor is not None:
        st.subheader("Model details")
        c1, c2, c3 = st.columns(3)
        c1.metric("Symptoms", len(predictor.symptoms))
        c2.metric("Diseases", len(predictor.diseases))
        c3.metric("Model type", "See metadata.json")

        meta_path = os.path.join(os.path.dirname(__file__), "models", "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            st.markdown(
                f"- **Best model:** {meta.get('best_model_name')}\n"
                f"- **Validation accuracy:** {meta.get('validation_accuracy', 0) * 100:.2f}%\n"
                f"- **Test accuracy:** {meta.get('test_accuracy', 0) * 100:.2f}%"
            )

    st.subheader("Technologies used")
    st.markdown(
        """
- **scikit-learn** — model training (Decision Tree, Random Forest,
  Logistic Regression, Naive Bayes, SVM)
- **pandas / numpy** — data processing
- **joblib** — model persistence
- **SQLite** — patient accounts and prediction history
- **pypdf** — reading uploaded report PDFs in the chatbot
- **Streamlit** — this frontend
        """
    )

    st.markdown(
        '<div class="disclaimer-box">⚠️ <b>Disclaimer:</b> This is an educational/'
        "academic project. Predictions are not a substitute for professional "
        "medical advice, diagnosis, or treatment.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Account (Login / Register / Logout)
# ---------------------------------------------------------------------------
elif page == "Account":
    st.title("🔐 Account")

    if st.session_state.user:
        st.success(f"You're logged in as **{st.session_state.user['username']}** ({st.session_state.user['email']}).")
        if st.button("Log out", type="primary"):
            st.session_state.user = None
            st.session_state.chat_messages = []
            st.rerun()
    else:
        login_tab, register_tab = st.tabs(["Log in", "Create account"])

        with login_tab:
            with st.form("login_form"):
                identifier = st.text_input("Username or email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in", type="primary")

            if submitted:
                if not identifier or not password:
                    st.warning("Please enter both fields.")
                else:
                    user = authenticate_user(identifier, password)
                    if user is None:
                        st.error("Incorrect username/email or password.")
                    else:
                        st.session_state.user = user
                        st.session_state.page = "Home"
                        st.success(f"Welcome back, {user['username']}!")
                        st.rerun()

        with register_tab:
            with st.form("register_form"):
                new_username = st.text_input("Username", key="reg_username")
                new_email = st.text_input("Email", key="reg_email")
                new_password = st.text_input("Password", type="password", key="reg_password")
                confirm_password = st.text_input("Confirm password", type="password", key="reg_confirm")
                agree = st.checkbox("I understand this app does not provide medical diagnosis.")
                reg_submitted = st.form_submit_button("Create account", type="primary")

            if reg_submitted:
                if new_password != confirm_password:
                    st.error("Passwords don't match.")
                elif not agree:
                    st.warning("Please acknowledge the disclaimer to continue.")
                else:
                    try:
                        user_id = register_user(new_username, new_email, new_password)
                        st.success("Account created! Please log in from the 'Log in' tab.")
                    except ValidationError as e:
                        st.error(str(e))
                    except UserAlreadyExistsError as e:
                        st.error(str(e))

