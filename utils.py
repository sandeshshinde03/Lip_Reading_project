import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

def extract_frames(video_path, output_dir, max_frames=30, min_frames=5):
    """
    Extract frames from a video with validation checks
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count < min_frames:
        print(f"Warning: {video_path} has only {frame_count} frames (min {min_frames} required)")
        cap.release()
        return None

    frames = []
    for i in range(min(frame_count, max_frames)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        frames.append(frame)
        
        # Save frame
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:03d}.jpg"), 
                   cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    cap.release()
    return np.array(frames)

def extract_mouth_roi(frames):
    """
    Extract mouth region with improved landmark detection
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    
    mouth_frames = []
    for frame in frames:
        results = face_mesh.process(frame)
        if not results.multi_face_landmarks:
            continue
            
        # Get mouth landmarks
        h, w = frame.shape[:2]
        mouth_landmarks = []
        for idx in [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 
                   178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 
                   321, 324, 375, 402, 405, 409, 415]:
            landmark = results.multi_face_landmarks[0].landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            mouth_landmarks.append((x, y))
        
        if mouth_landmarks:
            # Calculate bounding box with 20% padding
            x_coords = [p[0] for p in mouth_landmarks]
            y_coords = [p[1] for p in mouth_landmarks]
            
            x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
            
            # Add padding
            width = x_max - x_min
            height = y_max - y_min
            padding = int(max(width, height) * 0.2)
            
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)
            
            mouth_roi = frame[y_min:y_max, x_min:x_max]
            if mouth_roi.size > 0:  # Check if ROI is valid
                mouth_roi = cv2.resize(mouth_roi, (64, 64))
                mouth_frames.append(mouth_roi)
    
    face_mesh.close()
    return np.array(mouth_frames) if mouth_frames else None

def load_data(data_dir, max_samples_per_class=50, max_frames=30, min_frames=5):
    """
    Load data with proper sequence padding and validation
    """
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not classes:
        raise ValueError(f"No valid class folders found in {data_dir}")
    
    label_map = {label: i for i, label in enumerate(classes)}
    X = []
    y = []
    frame_counts = []
    
    print(f"\nFound {len(classes)} classes: {classes}")
    
    for label in classes:
        class_dir = os.path.join(data_dir, label)
        video_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))][:max_samples_per_class]
        
        if not video_files:
            print(f"Warning: No videos found in {class_dir}")
            continue
            
        print(f"\nProcessing {label} ({len(video_files)} videos):")
        
        for video_file in tqdm(video_files, desc=label):
            video_path = os.path.join(class_dir, video_file)
            frame_dir = os.path.join("data/frames/train", label, os.path.splitext(video_file)[0])
            
            # Extract frames
            frames = extract_frames(video_path, frame_dir, max_frames, min_frames)
            if frames is None:
                continue
                
            # Extract mouth ROIs
            mouth_frames = extract_mouth_roi(frames)
            if mouth_frames is None or len(mouth_frames) < min_frames:
                print(f"Skipping {video_file} - insufficient mouth frames")
                continue
                
            X.append(mouth_frames)
            y.append(label_map[label])
            frame_counts.append(len(mouth_frames))
    
    if not X:
        raise ValueError("No valid training samples found after processing")
    
    # Pad sequences to uniform length
    max_len = max(frame_counts) if frame_counts else max_frames
    print(f"\nMaximum sequence length: {max_len} frames")
    
    X_padded = np.zeros((len(X), max_len, 64, 64, 3))
    for i, seq in enumerate(X):
        X_padded[i, :len(seq)] = seq
    
    # Convert to numpy arrays
    X_final = X_padded / 255.0  # Normalize
    y_final = np.array(y)
    
    print(f"\nFinal dataset shape: {X_final.shape}")
    print(f"Class distribution: {np.bincount(y_final)}")
    
    return X_final, y_final, label_map

def preprocess_data(X, y, num_classes, test_size=0.2):
    """
    Handle classes with insufficient samples
    """
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    
    # Check class distribution
    class_counts = np.bincount(y)
    print("\nClass distribution before splitting:", class_counts)
    
    # Identify classes with insufficient samples
    min_samples = 2  # Minimum samples needed per class
    valid_classes = [i for i, count in enumerate(class_counts) if count >= min_samples]
    
    if len(valid_classes) < num_classes:
        print(f"Warning: Only {len(valid_classes)}/{num_classes} classes have ≥{min_samples} samples")
        
    # Filter samples from valid classes only
    mask = np.isin(y, valid_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    if len(X_filtered) == 0:
        raise ValueError("No classes have sufficient samples for splitting")
    
    # One-hot encode
    y_categorical = to_categorical(y_filtered, num_classes=num_classes)
    
    # Split data (without stratification if classes are imbalanced)
    X_train, X_val, y_train, y_val = train_test_split(
        X_filtered, y_categorical,
        test_size=test_size,
        random_state=42,
        shuffle=True
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    return X_train, X_val, y_train, y_val
    """
    Improved preprocessing with stratification
    """
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    
    # One-hot encode labels
    y_categorical = to_categorical(y, num_classes=num_classes)
    
    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, 
        test_size=test_size, 
        stratify=y,
        random_state=42
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val