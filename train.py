import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv3D, MaxPooling3D, TimeDistributed, 
                                    Flatten, Dense, Dropout, LSTM, 
                                    BatchNormalization)
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import load_data
import os
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

# Constants
MODEL_PATH = "models/lip_reading_model.h5"
MAX_FRAMES = 30
MOUTH_SHAPE = (64, 64, 3)

class VideoDataGenerator(Sequence):
    """Video data generator with proper augmentation"""
    def __init__(self, X, y, batch_size=8, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.05,
            zoom_range=0.05,
            fill_mode='constant',
            cval=0
        )
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_X = self.X[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]
        return batch_X, batch_y
    
    def on_epoch_end(self):
        if self.augment:
            indices = np.arange(len(self.X))
            np.random.shuffle(indices)
            self.X = self.X[indices]
            self.y = self.y[indices]

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv3D(32, (3,3,3), activation='relu', 
              input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling3D((1,2,2)),
        Dropout(0.3),
        
        Conv3D(64, (3,3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling3D((1,2,2)),
        Dropout(0.4),
        
        TimeDistributed(Flatten()),
        
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(32),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def preprocess_data(X, y, num_classes):
    """Handle small datasets properly"""
    # Convert to one-hot if needed
    if len(y.shape) == 1 or y.shape[1] != num_classes:
        y = to_categorical(y, num_classes=num_classes)
    
    # For very small datasets, use all data for training
    if len(X) < 50:  # Threshold for small dataset
        print("⚠️ Small dataset detected - using all samples for training")
        return X, X[:0], y, y[:0]  # Empty validation set
    
    # For larger datasets, do proper split
    from sklearn.model_selection import train_test_split
    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

def train():
    # Load data
    X, y, label_map = load_data("data/train_videos")
    num_classes = len(label_map)
    
    # Save label map
    os.makedirs("models", exist_ok=True)
    with open("models/label_map.pkl", "wb") as f:
        pickle.dump(label_map, f)
    
    # Preprocess data
    X_train, X_val, y_train, y_val = preprocess_data(X, y, num_classes)
    
    # Check class distribution
    class_counts = np.sum(y_train, axis=0) if len(y_train) > 0 else np.zeros(num_classes)
    print("\nClass distribution:", {k:int(v) for k,v in zip(label_map.keys(), class_counts)})
    
    # Handle small datasets with K-Fold
    if len(X_train) < 50 or len(X_val) == 0:
        print("\n⚠️ Using K-Fold validation for small dataset")
        kf = KFold(n_splits=min(3, len(X_train)), shuffle=True)
        best_val_acc = 0
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
            print(f"\nTraining fold {fold}/{kf.get_n_splits()}")
            X_tr, X_v = X_train[train_idx], X_train[val_idx]
            y_tr, y_v = y_train[train_idx], y_train[val_idx]
            
            train_gen = VideoDataGenerator(X_tr, y_tr, batch_size=4, augment=True)
            val_gen = VideoDataGenerator(X_v, y_v, batch_size=4)
            
            model = create_model(X_tr.shape[1:], num_classes)
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=30,
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Track best fold
            current_val_acc = max(history.history['val_accuracy'])
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                model.save(MODEL_PATH)
                print(f"Fold {fold} achieved new best val accuracy: {best_val_acc:.4f}")
    else:
        # Normal training for larger datasets
        train_gen = VideoDataGenerator(X_train, y_train, batch_size=8, augment=True)
        val_gen = VideoDataGenerator(X_val, y_val, batch_size=8)
        
        model = create_model(X_train.shape[1:], num_classes)
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=50,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    MODEL_PATH,
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                tf.keras.callbacks.EarlyStopping(patience=10)
            ]
        )
    
    print(f"\nTraining completed. Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/frames/train", exist_ok=True)
    train()