import streamlit as st
import os
import cv2
import numpy as np
from utils import extract_frames, extract_mouth_roi
import tensorflow as tf
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import base64

# Constants
MAX_FRAMES = 30
MOUTH_SHAPE = (64, 64, 3)
FRAMES_PER_ROW = 5

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("models/lip_reading_model.h5")
        with open("models/label_map.pkl", "rb") as f:
            label_map = pickle.load(f)
        reverse_label_map = {v: k for k, v in label_map.items()}
        return model, reverse_label_map
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

# Show debug info
def show_debug_info(video_path, processed_frames):
    st.subheader("Debug Information")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    tab1, tab2 = st.tabs(["Video Info", "Frame Analysis"])

    with tab1:
        st.write(f"📁 Video: `{os.path.basename(video_path)}`")
        st.write(f"🎞️ FPS: `{fps:.2f}`")
        st.write(f"🧮 Total Frames: `{frame_count}`")
        st.write(f"📊 Processed Frames: `{len(processed_frames)}/{MAX_FRAMES}`")

    with tab2:
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(processed_frames)), [np.mean(frame) for frame in processed_frames])
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Mean Pixel Value")
        ax.set_title("Frame Intensity Over Time")
        st.pyplot(fig)

# Preprocess video
def preprocess_video(video_path):
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join("data/frames/test", video_name)
        os.makedirs(output_dir, exist_ok=True)

        frames = extract_frames(video_path, output_dir, MAX_FRAMES)
        if frames is None or len(frames) < 5:
            st.warning(f"❗ Insufficient frames: {len(frames) if frames else 0}")
            return None

        mouth_frames = extract_mouth_roi(frames)
        if mouth_frames is None or len(mouth_frames) < 5:
            st.warning(f"❗ Insufficient mouth frames: {len(mouth_frames) if mouth_frames else 0}")
            return None

        X = np.zeros((1, MAX_FRAMES, *MOUTH_SHAPE))
        valid_frames = min(len(mouth_frames), MAX_FRAMES)
        X[0, :valid_frames] = mouth_frames[:valid_frames]

        expected_shape = (1, MAX_FRAMES, *MOUTH_SHAPE)
        if X.shape != expected_shape:
            st.error(f"🚫 Shape mismatch: {X.shape} vs expected {expected_shape}")
            return None

        return X / 255.0

    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
        return None

# Main app
def main():
    st.set_page_config(page_title="Lip Reading App", layout="wide")
    st.markdown("""
        <style>
        .main {
            background-color: #f4f4f8;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .video-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .video-container video {
            width: 480px;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("👄 Lip Reading App")
    st.markdown("""
    Upload a short video of a person speaking. The model will predict the **spoken word** based on lip movements.  
    Supported formats: `.mp4`, `.mov`, `.avi`  
    Tip: Use close-up videos for accurate results.
    """)

    st.sidebar.title("⚙️ Settings")
    debug_mode = st.sidebar.checkbox("Enable Debug Mode")

    model, reverse_label_map = load_model()
    if model is None:
        return

    uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "mov", "avi"])

    if uploaded_file:
        try:
            os.makedirs("data/test_videos", exist_ok=True)
            video_path = os.path.join("data/test_videos", uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Display styled video in center with fixed size using base64
            video_bytes = uploaded_file.getvalue()
            video_base64 = base64.b64encode(video_bytes).decode()
            st.markdown(f"""
                <div class="video-container">
                    <video controls>
                        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
            """, unsafe_allow_html=True)

            with st.spinner("⏳ Processing..."):
                X = preprocess_video(video_path)
                if X is None:
                    return


                if X.shape[1:] != model.input_shape[1:]:
                    st.error(f"⚠️ Shape mismatch: {X.shape[1:]} vs model expected {model.input_shape[1:]}")
                    return

                pred = model.predict(X)
                pred_class = np.argmax(pred[0])
                confidence = np.max(pred[0])

                # Big predicted word display
                st.markdown(f"""
                    <div style='
                        font-size: 20px; 
                        font-weight: bold; 
                        color: #2E8B57; 
                        padding: 20px; 
                        border: 3px solid #4CAF50; 
                        border-radius: 12px; 
                        background-color: #e6ffe6; 
                        text-align: center;
                        margin-top: 30px;
                    '>
                        🔮 Predicted Word: {reverse_label_map[pred_class]} 
                    </div>
                """, unsafe_allow_html=True)

                if debug_mode:
                    show_debug_info(video_path, X[0])

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    os.makedirs("data/test_videos", exist_ok=True)
    os.makedirs("data/frames/test", exist_ok=True)
    main()
