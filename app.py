import os
import cv2
import time
import pickle
import random
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
from Face_recognition_model import (
    build_detector, detect_face, face_to_tensor,
    Conv2D, MaxPool2D, BatchNorm, Dense, Dropout, 
    FaceRecognitionCNN, Adam, IMG_SIZE
)

st.set_page_config(page_title="Face Recognition System", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
        color: white;
    }
    .block-container {
        padding-top: 2rem;
    }
    .title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
        color: white;
    }
    .subtitle {
        text-align: center;
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-bottom: 20px;
    }
    .status-box {
        background: rgba(50, 100, 150, 0.15);
        border: 1px solid rgba(100, 180, 220, 0.3);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .card {
        background: rgba(255,255,255,0.08);
        padding: 24px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .menu-option {
        background: rgba(100, 150, 200, 0.1);
        border: 1px solid rgba(100, 150, 200, 0.3);
        padding: 10px;
        margin: 8px 0;
        border-radius: 8px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>Face Recognition System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enrollment • Training • Live Recognition</div>", unsafe_allow_html=True)

MODEL_PATH = "face_model.pkl"
DATA_DIR = "enrolled_faces"

os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return FaceRecognitionCNN.load(MODEL_PATH)
    return None

def reload_model():
    """Force reload the model (bypassing cache)"""
    st.cache_resource.clear()
    if os.path.exists(MODEL_PATH):
        return FaceRecognitionCNN.load(MODEL_PATH)
    return None

@st.cache_resource
def build_det():
    return build_detector()

def get_enrolled_people():
    """Get list of enrolled people from disk."""
    people = []
    if os.path.exists(DATA_DIR):
        for person_dir in os.listdir(DATA_DIR):
            person_path = os.path.join(DATA_DIR, person_dir)
            if os.path.isdir(person_path):
                face_files = [f for f in os.listdir(person_path) if f.endswith('.npy')]
                people.append((person_dir, len(face_files)))
    return people

def get_status():
    """Return enrollment and model status."""
    people = get_enrolled_people()
    enrolled_count = len(people)
    model_trained = os.path.exists(MODEL_PATH)
    return {
        'enrolled_count': enrolled_count,
        'people': people,
        'model_trained': model_trained
    }

def display_status():
    """Display status dashboard."""
    status = get_status()
    enrolled = status['enrolled_count']
    trained = "✓ trained" if status['model_trained'] else "✗ not trained"
    
    status_text = f"""
────────────────────────────────────────────────────
  Enrolled      : {enrolled} person{"s" if enrolled != 1 else ""}
  Model Status  : {trained}
────────────────────────────────────────────────────"""
    
    st.markdown(f"<div class='status-box'><pre>{status_text}</pre></div>", unsafe_allow_html=True)
    return status

with st.sidebar:
    selected = option_menu(
        menu_title="📋 Dashboard",
        options=["Home", "Enroll", "Train", "Recognize", "Manage", "Info"],
        icons=["house", "person-plus", "cpu", "camera", "gear", "info-circle"],
        default_index=0,
    )

detector = build_det()
model = load_model()

if selected == "Home":
    status = display_status()
    
    col1, col2 = st.columns([1.1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🚀 Quick Start")
        st.write(
            "**Face Recognition System** with custom CNN"
        )
        st.markdown("""
1. **Enroll** new people using camera capture
2. **Train** the model on enrolled faces (minimum 1 person)
3. **Recognize** faces in real-time or from uploaded images
4. **Manage** models and enrolled data
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.image(
            "https://images.unsplash.com/photo-1526379095098-d400fd0bf935?q=80&w=1200&auto=format&fit=crop",
            use_container_width=True,
        )

elif selected == "Enroll":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👤 Enroll New Person")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        person_name = st.text_input("Person Name:", placeholder="Enter full name")
        
        tab1, tab2 = st.tabs(["Upload Image", "Webcam Capture"])
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Upload face image (JPG/PNG)",
                type=["jpg", "jpeg", "png"]
            )
            
            if uploaded_file and person_name:
                image = Image.open(uploaded_file).convert("RGB")
                frame = np.array(image)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                face = detect_face(gray, detector)
                
                if face is None:
                    st.warning("❌ No face detected in image")
                else:
                    st.success("✅ Face detected!")
                    x, y, w, h = face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    st.image(frame, channels="RGB", use_container_width=True)
                    
                    if st.button("Save Face", key="save_face_upload"):
                        person_dir = os.path.join(DATA_DIR, person_name)
                        os.makedirs(person_dir, exist_ok=True)
                        
                        face_tensor = face_to_tensor(gray, face)
                        face_count = len(os.listdir(person_dir))
                        face_path = os.path.join(person_dir, f"face_{face_count}.npy")
                        np.save(face_path, face_tensor)
                        
                        st.success(f"✅ Face saved for {person_name}!")
                        st.rerun()
        
        with tab2:
            camera_input = st.camera_input("Take a photo from webcam")
            
            if camera_input and person_name:
                image = Image.open(camera_input).convert("RGB")
                frame = np.array(image)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                face = detect_face(gray, detector)
                
                if face is None:
                    st.warning("❌ No face detected in image")
                else:
                    st.success("✅ Face detected!")
                    x, y, w, h = face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    st.image(frame, channels="RGB", use_container_width=True)
                    
                    if st.button("Save Face", key="save_face_camera"):
                        person_dir = os.path.join(DATA_DIR, person_name)
                        os.makedirs(person_dir, exist_ok=True)
                        
                        face_tensor = face_to_tensor(gray, face)
                        face_count = len(os.listdir(person_dir))
                        face_path = os.path.join(person_dir, f"face_{face_count}.npy")
                        np.save(face_path, face_tensor)
                        
                        st.success(f"✅ Face saved for {person_name}!")
                        st.rerun()
    
    with col2:
        status = display_status()
        st.subheader("Enrolled People")
        if status['people']:
            for name, count in status['people']:
                st.info(f"👤 **{name}**: {count} face(s)")
        else:
            st.info("No people enrolled yet")
    
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Train":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎓 Train Model")
    
    status = display_status()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Training Parameters**")
        epochs = st.slider("Epochs", 5, 50, 20, step=5)
        lr = st.select_slider("Learning Rate", 
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001)
        batch_size = st.slider("Batch Size", 4, 32, 16, step=4)
        
        if st.button("🚀 Start Training", key="train_btn"):
            if status['enrolled_count'] < 1:
                st.error("❌ Need at least 1 enrolled person to train!")
            else:
                X_train, y_train, label_names = [], [], []
                
                for person_dir in os.listdir(DATA_DIR):
                    person_path = os.path.join(DATA_DIR, person_dir)
                    if os.path.isdir(person_path):
                        label_names.append(person_dir)
                        for face_file in os.listdir(person_path):
                            if face_file.endswith('.npy'):
                                face = np.load(os.path.join(person_path, face_file))
                                X_train.append(face[0])  
                                y_train.append(len(label_names) - 1)
                
                if not X_train:
                    st.error("❌ No face data found!")
                else:
                    X_train = np.array(X_train).astype(np.float32)  
                    X_train = X_train[:, np.newaxis, :, :] 
                    y_train = np.array(y_train)
                    
                    # Always create a fresh model with correct number of classes
                    # This ensures old models with different class counts are replaced
                    model = FaceRecognitionCNN(num_classes=len(label_names))
                    model.label_names = label_names
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    st.write(f"**Training on {len(X_train)} faces from {len(label_names)} people...**")
                    st.write(f"**People:** {', '.join(label_names)}")
                    
                    model.train(X_train, y_train, epochs=epochs, lr=lr, batch=batch_size)
                    
                    # Save the model
                    model.save(MODEL_PATH)
                    
                    # Clear cache to force reload of new model on next run
                    st.cache_resource.clear()
                    
                    progress_bar.progress(100)
                    st.success("✅ Model trained and saved!")
                    st.rerun()
    
    with col2:
        if status['people']:
            st.subheader("📊 Training Data")
            total_faces = sum(count for _, count in status['people'])
            st.metric("People", status['enrolled_count'])
            st.metric("Total Faces", total_faces)
            
            for name, count in status['people']:
                st.info(f"👤 **{name}**: {count} faces")
        else:
            st.warning("⚠️ No enrolled people yet. Enroll faces first!")
    
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Recognize":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Recognize Faces")
    
    if model is None:
        st.error("❌ Model not trained! Please train first.")
    else:
        tab1, tab2 = st.tabs(["Upload Image", "Webcam"])
        
        with tab1:
            uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
            
            if uploaded is not None:
                image = Image.open(uploaded).convert("RGB")
                frame = np.array(image)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                face = detect_face(gray, detector)
                
                if face is None:
                    st.warning("⚠️ No face detected")
                else:
                    tensor = face_to_tensor(gray, face)
                    pred, confidence, name = model.predict_single(tensor, confidence_threshold=0.55)
                    confidence = confidence * 100
                    
                    x, y, w, h = face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    if name == "Unknown":
                        cv2.putText(
                            frame,
                            f"Unknown {confidence:.1f}%",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )
                        st.image(frame, channels="RGB", use_container_width=True)
                        st.warning(f"⚠️ **Unknown** ({confidence:.1f}% - below confidence threshold)")
                    else:
                        cv2.putText(
                            frame,
                            f"{name} {confidence:.1f}%",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                        st.image(frame, channels="RGB", use_container_width=True)
                        st.success(f"✅ **{name}** ({confidence:.1f}%)")
        
        with tab2:
            camera_input = st.camera_input("Capture from webcam")
            
            if camera_input is not None:
                image = Image.open(camera_input).convert("RGB")
                frame = np.array(image)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                face = detect_face(gray, detector)
                
                if face is None:
                    st.warning("⚠️ No face detected")
                else:
                    tensor = face_to_tensor(gray, face)
                    pred, confidence, name = model.predict_single(tensor, confidence_threshold=0.55)
                    confidence = confidence * 100
                    
                    x, y, w, h = face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    if name == "Unknown":
                        cv2.putText(
                            frame,
                            f"Unknown {confidence:.1f}%",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )
                        st.image(frame, channels="RGB", use_container_width=True)
                        st.warning(f"⚠️ **Unknown** ({confidence:.1f}% - below confidence threshold)")
                    else:
                        cv2.putText(
                            frame,
                            f"{name} {confidence:.1f}%",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                        st.image(frame, channels="RGB", use_container_width=True)
                        st.success(f"✅ **{name}** ({confidence:.1f}%)")
    
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Manage":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Manage Model & Data")
    
    status = display_status()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("💾 Model Operations")
        
        if st.button("💾 Save Current Model", key="save_model"):
            if model:
                model.save(MODEL_PATH)
                st.success("✅ Model saved!")
            else:
                st.error("❌ No model to save")
        
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) // 1024
            st.info(f"📁 Model file: {file_size} KB")
    
    with col2:
        st.subheader("🗑️ Delete Data")
        
        if status['people']:
            person_to_delete = st.selectbox(
                "Select person to delete:",
                [name for name, _ in status['people']]
            )
            
            if st.button("🗑️ Delete Person", key="delete_person"):
                person_path = os.path.join(DATA_DIR, person_to_delete)
                import shutil
                shutil.rmtree(person_path)
                st.success(f"✅ Deleted {person_to_delete}")
                st.rerun()
        else:
            st.info("No enrolled people to delete")
    
    st.subheader("📊 Data Summary")
    st.write(f"**Total People:** {status['enrolled_count']}")
    if status['people']:
        total_faces = sum(count for _, count in status['people'])
        st.write(f"**Total Faces:** {total_faces}")
        
        st.write("**Breakdown:**")
        for name, count in status['people']:
            st.write(f"  • {name}: {count} faces")
    
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Info":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("ℹ️ Model Information")
    
    if model is not None and hasattr(model, "label_names"):
        st.metric("Registered Classes", len(model.label_names))
        
        st.write("**🎓 Recognized Labels:**")
        for idx, item in enumerate(model.label_names, 1):
            st.write(f"  {idx}. {item}")
        
        st.write("**📋 Model Architecture:**")
        st.markdown("""
- **Input:** 48×48 grayscale images
- **Layers:**
  - Conv2D (1→32) + BatchNorm + MaxPool
  - Conv2D (32→64) + BatchNorm + MaxPool
  - Conv2D (64→128) + BatchNorm + MaxPool
  - Dense (6144→256) + BatchNorm + Dropout(0.5)
  - Dense (256→num_classes) + Softmax
- **Optimizer:** Adam
- **Framework:** Pure NumPy (no TensorFlow/PyTorch)
        """)
    else:
        st.error("❌ Model not trained yet")
    
    st.markdown("</div>", unsafe_allow_html=True)
