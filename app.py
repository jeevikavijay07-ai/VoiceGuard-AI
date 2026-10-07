import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import os

st.set_page_config(
    page_title="VoiceGuard AI",
    page_icon="🎙️",
    layout="centered"
)

MODEL_FILE = "voiceguard_model.pkl"
SCALER_FILE = "voiceguard_scaler.pkl"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    delta = librosa.feature.delta(mfcc)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.std(delta, axis=1)
    ])

    return features


st.title("🎙️ VoiceGuard AI")
st.subheader("AI-Powered Voice Cloning Fraud Detection")

st.write(
    "VoiceGuard combines voice authenticity with call-context "
    "signals to estimate fraud risk before high-impact actions."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a WAV voice recording",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    st.subheader("📞 Call Context")

    caller_known = st.selectbox(
        "Is this a known/trusted caller?",
        ["Yes", "No"]
    )

    urgent_request = st.selectbox(
        "Is the caller making an urgent payment request?",
        ["No", "Yes"]
    )

    otp_request = st.selectbox(
        "Is the caller asking for OTP/PIN/password?",
        ["No", "Yes"]
    )

    transaction_amount = st.number_input(
        "Transaction amount (₹)",
        min_value=0,
        value=0,
        step=1000
    )

    if st.button("🔍 Analyze Voice & Risk"):

        with st.spinner("Analyzing voice and risk signals..."):

            model, scaler = load_model()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as temp_file:

                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            features = extract_features(temp_path)
            features = scaler.transform([features])

            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]

            real_probability = probability[0]
            fake_probability = probability[1]

            os.remove(temp_path)

        # Voice risk
        risk_score = fake_probability * 60

        # Context risk
        if caller_known == "No":
            risk_score += 10

        if urgent_request == "Yes":
            risk_score += 10

        if otp_request == "Yes":
            risk_score += 15

        if transaction_amount >= 50000:
            risk_score += 5

        risk_score = min(risk_score, 100)

        if risk_score >= 70:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        st.divider()

        st.subheader("🎯 Detection Result")

        if prediction == 1:
            st.error("⚠️ FAKE / AI-GENERATED VOICE")
        else:
            st.success("✅ LIKELY REAL VOICE")

        st.metric(
            "Fake Probability",
            f"{fake_probability * 100:.2f}%"
        )

        st.divider()

        st.subheader("🚨 Fraud Risk Assessment")

        if risk_level == "HIGH":
            st.error(f"🔴 HIGH RISK — {risk_score:.0f}/100")
        elif risk_level == "MEDIUM":
            st.warning(f"🟡 MEDIUM RISK — {risk_score:.0f}/100")
        else:
            st.success(f"🟢 LOW RISK — {risk_score:.0f}/100")

        st.write("**Risk factors detected:**")

        if fake_probability >= 0.5:
            st.write("🔊 Voice authenticity concern")

        if caller_known == "No":
            st.write("📞 Unknown/untrusted caller")

        if urgent_request == "Yes":
            st.write("⚡ Urgent payment request")

        if otp_request == "Yes":
            st.write("🔐 OTP/PIN request")

        if transaction_amount >= 50000:
            st.write("💰 High-value transaction")

        st.divider()

        st.subheader("🛡️ Recommended Next Step")

        if risk_level == "HIGH":
            st.error(
                "Additional verification required. "
                "For high-impact actions, escalate to human review "
                "or verify through a trusted channel."
            )

        elif risk_level == "MEDIUM":
            st.warning(
                "Verify the caller's identity through an independent "
                "trusted channel before proceeding."
            )

        else:
            st.success(
                "Continue with caution. Voice authenticity alone "
                "should not be treated as proof."
            )

        st.caption(
            "Prototype risk score for demonstration. "
            "Voice probability is not proof of fraud. "
            "High-impact decisions should use additional verification."
        )