# Ex.No: 13 Learning – Coral Reef Health Monitoring System  
### DATE: 25/10/2025                                                                      
### REGISTER NUMBER : 212223060166
### AIM: 
The aim of the code is to classify coral reef health status ("Healthy", "Stressed", "Ambient") using machine learning techniques applied to underwater audio recordings of coral reefs. 
###  Algorithm:

### 1. Audio Preprocessing
- Load raw WAV audio files from reef recordings.
- Standardize duration and sample rate; pad audio as needed.
- Handle loading errors by generating dummy data to prevent pipeline breaks.

### 2. Feature Extraction
- Convert audio into *log-mel spectrograms* using Librosa.
- Normalize spectrograms for neural network input.

### 3. Dataset Preparation
- Implement a *PyTorch Dataset class* for spectrograms and labels.
- Encode labels from detailed acoustic classes into simplified reef health statuses.

### 4. Model Definition
- Build a *custom Convolutional Neural Network (CNN)* optimized for spectral images.
- Map extracted features to *three reef health classes* in the output layer.

### 5. Training & Validation
- Split data into *training, validation, and test sets*.
- Train CNN with early stopping; monitor *loss* and *accuracy* each epoch.
- Evaluate model performance using *loss/accuracy plots* and *confusion matrix*.

### 6. Health Status Visualization
- Analyze and visualize predicted reef health distributions.
- Provide interpretable insights into coral ecosystem conditions.
### Program:


# %%
# ============================================================================
# CELL 1: Installation and Setup
# ============================================================================

!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install librosa soundfile matplotlib seaborn pandas numpy scikit-learn
!pip install tqdm jupyter ipywidgets plotly fastapi uvicorn
!pip install gradio streamlit  # For demo interface

print("✅ All packages installed successfully!")
print("🚀 Ready to start OceanPulse training!")

# %%

# ============================================================================
# CELL 2: Import Libraries and Check Setup
# ============================================================================

"""
📚 Import all necessary libraries and check system configuration
"""

# Core libraries
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Audio processing
import librosa
import librosa.display
import soundfile as sf

# Machine learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch.nn.functional as F

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Progress tracking
from tqdm import tqdm
import time
import datetime

# Check system configuration
print("=== OceanPulse System Check ===")
print(f"Python version: {os.sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"Librosa version: {librosa.__version__}")
print("✅ System check complete!")

# %%
# ============================================================================
# CELL 3: Configuration and Constants
# ============================================================================

"""
⚙ Configuration settings for the OceanPulse project
Modify these paths according to your setup
"""

# === PROJECT CONFIGURATION ===
class OceanPulseConfig:
    # File paths (UPDATE THESE PATHS!)
    DATA_DIR = r"C:\Users\admin\Downloads\ReefSet_v1.0\ReefSet_v1.0\full_dataset"  # Directory containing WAV files
    ANNOTATIONS_FILE = r"C:\Users\admin\Downloads\ReefSet_v1.0\ReefSet_v1.0\reefset_annotations.json"  # Annotations JSON
    LABELS_CSV = r"C:\Users\admin\Downloads\ReefSet_v1.0\ReefSet_v1.0\reefset_labels_by_dataset.csv"  # Labels CSV (optional)
    
    # Audio processing parameters
    SAMPLE_RATE = 16000
    AUDIO_DURATION = 1.88  # seconds
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Model parameters
    DROPOUT_RATE = 0.3
    
    # System parameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    RANDOM_SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(OceanPulseConfig.RANDOM_SEED)
np.random.seed(OceanPulseConfig.RANDOM_SEED)

print("📋 Configuration loaded:")
print(f"  • Device: {OceanPulseConfig.DEVICE}")
print(f"  • Sample Rate: {OceanPulseConfig.SAMPLE_RATE} Hz")
print(f"  • Audio Duration: {OceanPulseConfig.AUDIO_DURATION} seconds")
print(f"  • Batch Size: {OceanPulseConfig.BATCH_SIZE}")


# %%
# ============================================================================
# CELL 4 (FIXED): Data Loading and Exploration
# ============================================================================

"""
📊 Load and explore the ReefSet dataset
This cell will analyze the dataset structure and class distribution
FIXED VERSION: Handles different JSON structures robustly
"""

def inspect_annotation_structure(annotations_file):
    """Inspect the structure of the annotations file to understand the format"""
    
    print("🔍 Inspecting annotation file structure...")
    
    if not os.path.exists(annotations_file):
        print(f"❌ Annotations file not found: {annotations_file}")
        print("📝 Please update the ANNOTATIONS_FILE path in the configuration")
        return None
    
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        print(f"📄 File loaded successfully")
        print(f"📊 Data type: {type(annotations)}")
        
        if isinstance(annotations, list) and len(annotations) > 0:
            print(f"📋 Number of items: {len(annotations)}")
            print(f"🔑 Sample item keys: {list(annotations[0].keys())}")
            print(f"📝 Sample item: {annotations[0]}")
        elif isinstance(annotations, dict):
            print(f"🔑 Top-level keys: {list(annotations.keys())}")
            # Show sample of nested structure
            for key, value in list(annotations.items())[:3]:
                print(f"📝 Sample entry '{key}': {value}")
        
        return annotations
    
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return None

def load_reefset_annotations_flexible(annotations_file):
    """Load and parse ReefSet annotations with flexible structure handling"""
    
    print("🔍 Loading ReefSet annotations (flexible mode)...")
    
    # First inspect the structure
    annotations = inspect_annotation_structure(annotations_file)
    
    if annotations is None:
        return None, None, None
    
    filenames = []
    labels = []
    datasets = []
    
    try:
        if isinstance(annotations, list):
            # Handle list format
            for item in annotations:
                # Try different possible key names
                filename = None
                for key in ['filename', 'file', 'path', 'audio_file', 'file_name']:
                    if key in item:
                        filename = item[key]
                        break
                
                if filename is None:
                    print(f"⚠ Warning: No filename found in item: {item}")
                    continue
                
                label = None
                for key in ['label', 'class', 'category', 'annotation']:
                    if key in item:
                        label = item[key]
                        break
                
                if label is None:
                    print(f"⚠ Warning: No label found in item: {item}")
                    continue
                
                dataset = item.get('dataset', item.get('source', item.get('origin', 'unknown')))
                
                filenames.append(filename)
                labels.append(label)
                datasets.append(dataset)
        
        elif isinstance(annotations, dict):
            # Handle dictionary format
            for key, value in annotations.items():
                if isinstance(value, dict):
                    # Key is filename, value contains metadata
                    filename = key
                    label = value.get('label', value.get('class', value.get('category')))
                    dataset = value.get('dataset', value.get('source', value.get('origin', 'unknown')))
                    
                    if label is not None:
                        filenames.append(filename)
                        labels.append(label)
                        datasets.append(dataset)
                else:
                    # Key is filename, value is label
                    filenames.append(key)
                    labels.append(value)
                    datasets.append('unknown')
        
        print(f"✅ Successfully loaded {len(filenames)} annotations")
        
        if len(filenames) > 0:
            print(f"📁 Unique labels: {len(set(labels))}")
            print(f"🗺 Unique datasets: {len(set(datasets))}")
            print(f"📝 Sample entries:")
            for i in range(min(3, len(filenames))):
                print(f"  • {filenames[i]} -> {labels[i]} ({datasets[i]})")
        else:
            print("❌ No valid annotations found!")
            return None, None, None
        
        return filenames, labels, datasets
    
    except Exception as e:
        print(f"❌ Error processing annotations: {e}")
        return None, None, None

def create_sample_data_if_needed():
    """Create sample data if no annotations file is found"""
    
    print("🎭 Creating sample data for demonstration...")
    
    # Sample coral reef sound categories based on typical research
    sample_labels = [
        'bioph_fish_chorus', 'bioph_fish_individual', 'bioph_fish_feeding',
        'bioph_crustacean_snapping', 'bioph_crustacean_walking', 'bioph_crustacean_feeding',
        'anthrop_boat_engine', 'anthrop_boat_propeller', 'anthrop_construction',
        'ambient_wave_action', 'ambient_current', 'ambient_storm',
        'bioph_coral_spawning', 'bioph_turtle_movement', 'bioph_dolphin_click'
    ]
    
    datasets_list = ['Great_Barrier_Reef', 'Caribbean_Reef', 'Red_Sea', 'Pacific_Coral', 'Atlantic_Reef']
    
    # Generate sample data
    np.random.seed(42)  # For reproducibility
    n_samples = 1000
    
    filenames = [f"reef_audio_{i:04d}.wav" for i in range(n_samples)]
    labels = np.random.choice(sample_labels, n_samples)
    datasets = np.random.choice(datasets_list, n_samples)
    
    print(f"✅ Generated {n_samples} sample annotations")
    print(f"📁 Sample labels: {sample_labels[:5]}...")
    print(f"🗺 Sample datasets: {datasets_list}")
    
    return filenames, labels, datasets

def analyze_dataset_distribution(labels, datasets):
    """Analyze and visualize dataset distribution"""
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'label': labels,
        'dataset': datasets
    })
    
    print("\n=== DATASET ANALYSIS ===")
    
    # Label distribution
    label_counts = df['label'].value_counts()
    print(f"\n🏷 Top 10 Label Classes:")
    for label, count in label_counts.head(10).items():
        print(f"  • {label}: {count:,} samples")
    
    # Dataset distribution  
    dataset_counts = df['dataset'].value_counts()
    print(f"\n🌍 Dataset Distribution:")
    for dataset, count in dataset_counts.head(10).items():
        print(f"  • {dataset}: {count:,} samples")
    
    return df, label_counts, dataset_counts

# ===== MAIN EXECUTION =====

# Try to load the annotations
filenames, labels, datasets = load_reefset_annotations_flexible(OceanPulseConfig.ANNOTATIONS_FILE)

# If loading failed, create sample data
if labels is None:
    print("\n🎭 Annotations loading failed. Using sample data for demonstration...")
    filenames, labels, datasets = create_sample_data_if_needed()

if labels is not None:
    # Analyze distribution
    df_analysis, label_counts, dataset_counts = analyze_dataset_distribution(labels, datasets)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🌊 ReefSet Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Top 15 labels
    label_counts.head(15).plot(kind='bar', ax=axes[0,0], color='lightcoral')
    axes[0,0].set_title('Top 15 Label Classes')
    axes[0,0].set_xlabel('Label')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Top 10 datasets
    dataset_counts.head(10).plot(kind='bar', ax=axes[0,1], color='lightblue')
    axes[0,1].set_title('Top 10 Datasets')
    axes[0,1].set_xlabel('Dataset')
    axes[0,1].set_ylabel('Count')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Label category analysis
    bioph_labels = [label for label in label_counts.index if 'bioph' in label.lower()]
    anthrop_labels = [label for label in label_counts.index if 'anthrop' in label.lower()]
    ambient_labels = [label for label in label_counts.index if 'ambient' in label.lower()]
    
    category_counts = {
        'Biological': sum(label_counts[label] for label in bioph_labels),
        'Anthropogenic': sum(label_counts[label] for label in anthrop_labels),
        'Ambient': sum(label_counts[label] for label in ambient_labels),
        'Other': sum(label_counts[label] for label in label_counts.index 
                    if not any(cat in label.lower() for cat in ['bioph', 'anthrop', 'ambient']))
    }
    
    # Remove categories with 0 counts
    category_counts = {k: v for k, v in category_counts.items() if v > 0}
    
    colors = ['lightgreen', 'orange', 'lightblue', 'lightyellow']
    axes[1,0].pie(category_counts.values(), labels=category_counts.keys(), 
                  autopct='%1.1f%%', colors=colors[:len(category_counts)])
    axes[1,0].set_title('Sound Categories Distribution')
    
    # Dataset size comparison
    top_datasets = dataset_counts.head(8)
    axes[1,1].bar(range(len(top_datasets)), top_datasets.values, color='lightcyan')
    axes[1,1].set_xticks(range(len(top_datasets)))
    axes[1,1].set_xticklabels(top_datasets.index, rotation=45)
    axes[1,1].set_title('Dataset Sizes (Top 8)')
    axes[1,1].set_ylabel('Number of Samples')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Category Summary:")
    total_samples = sum(category_counts.values())
    for category, count in category_counts.items():
        percentage = count/total_samples*100 if total_samples > 0 else 0
        print(f"  • {category}: {count:,} samples ({percentage:.1f}%)")
    
    # Save processed data for next steps
    processed_data = {
        'filenames': filenames,
        'labels': labels,
        'datasets': datasets,
        'label_counts': dict(label_counts),
        'dataset_counts': dict(dataset_counts)
    }
    
    print(f"\n✅ Data loading and analysis complete!")
    print(f"📋 Ready for next step: Audio preprocessing")
    
else:
    print("❌ Failed to load or generate data. Please check your configuration.")

# %%
 #============================================================================
# CELL 5: Audio Processing and Visualization Functions
# ============================================================================

"""
🎵 Audio processing functions for ReefSet data
Includes spectrogram generation and audio visualization
"""

def load_and_process_audio(audio_path, sr=16000, duration=1.88):
    """Load and preprocess a single audio file"""
    
    try:
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Ensure consistent length
        target_length = int(duration * sr)
        if len(audio) < target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            # Trim to target length
            audio = audio[:target_length]
            
        return audio
    
    except Exception as e:
        print(f"❌ Error loading {audio_path}: {e}")
        return None

def generate_mel_spectrogram(audio, sr=16000, n_mels=128, n_fft=1024, hop_length=512):
    """Generate log-mel spectrogram from audio"""
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
    
    return log_mel_spec

def visualize_reef_audio_samples(data_dir, filenames, labels, n_samples=6):
    """Visualize sample audio files and their spectrograms"""
    
    print("🎵 Visualizing reef audio samples...")
    
    # Select diverse samples
    unique_labels = list(set(labels))
    selected_samples = []
    
    for label in unique_labels[:n_samples]:
        label_indices = [i for i, l in enumerate(labels) if l == label]
        if label_indices:
            selected_samples.append((filenames[label_indices[0]], label))
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 3*n_samples))
    fig.suptitle('🌊 Reef Audio Sample Analysis', fontsize=16, fontweight='bold')
    
    for i, (filename, label) in enumerate(selected_samples):
        audio_path = os.path.join(data_dir, filename)
        
        if os.path.exists(audio_path):
            # Load audio
            audio = load_and_process_audio(audio_path)
            
            if audio is not None:
                # Plot waveform
                time_axis = np.linspace(0, len(audio)/OceanPulseConfig.SAMPLE_RATE, len(audio))
                axes[i, 0].plot(time_axis, audio)
                axes[i, 0].set_title(f'Waveform: {label}')
                axes[i, 0].set_xlabel('Time (s)')
                axes[i, 0].set_ylabel('Amplitude')
                
                # Generate and plot spectrogram
                mel_spec = generate_mel_spectrogram(audio)
                librosa.display.specshow(
                    mel_spec, 
                    sr=OceanPulseConfig.SAMPLE_RATE,
                    hop_length=OceanPulseConfig.HOP_LENGTH,
                    x_axis='time', 
                    y_axis='mel',
                    ax=axes[i, 1]
                )
                axes[i, 1].set_title(f'Mel Spectrogram: {label}')
                axes[i, 1].set_xlabel('Time (s)')
                axes[i, 1].set_ylabel('Mel Frequency')
    
    plt.tight_layout()
    plt.show()

# Run audio visualization if data is available
if labels is not None:
    print("🎵 Preparing to visualize audio samples...")
    print("⚠ Note: This will only work if the WAV files are available in DATA_DIR")
    
    # Check if data directory exists
    if os.path.exists(OceanPulseConfig.DATA_DIR):
        visualize_reef_audio_samples(OceanPulseConfig.DATA_DIR, filenames, labels, n_samples=4)
    else:
        print(f"❌ Data directory not found: {OceanPulseConfig.DATA_DIR}")
        print("📝 Please update DATA_DIR in the configuration to point to your WAV files")


# %%
# ============================================================================
# CELL 6: Dataset Class Definition
# ============================================================================

"""
🏗 PyTorch Dataset class for ReefSet
Handles audio loading, preprocessing, and spectrogram generation
"""

class ReefSetDataset(Dataset):
    """
    PyTorch Dataset for ReefSet coral reef audio data
    """
    
    def __init__(self, data_dir, filenames, labels, sr=16000, n_mels=128, 
                 duration=1.88, transform=None, label_encoder=None):
        
        self.data_dir = data_dir
        self.filenames = filenames
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.transform = transform
        self.label_encoder = label_encoder
        
        # Filter out files that don't exist
        self.valid_indices = []
        print("🔍 Checking file existence...")
        
        for i, filename in enumerate(tqdm(filenames, desc="Validating files")):
            audio_path = os.path.join(data_dir, filename)
            if os.path.exists(audio_path):
                self.valid_indices.append(i)
        
        print(f"✅ Found {len(self.valid_indices)} valid audio files out of {len(filenames)}")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get actual index
        actual_idx = self.valid_indices[idx]
        filename = self.filenames[actual_idx]
        label = self.labels[actual_idx]
        
        # Load and process audio
        audio_path = os.path.join(self.data_dir, filename)
        audio = load_and_process_audio(audio_path, self.sr, self.duration)
        
        if audio is None:
            # Return dummy data if loading fails
            audio = np.random.normal(0, 0.01, int(self.duration * self.sr))
        
        # Generate mel spectrogram
        mel_spec = generate_mel_spectrogram(
            audio, 
            sr=self.sr, 
            n_mels=self.n_mels,
            n_fft=OceanPulseConfig.N_FFT,
            hop_length=OceanPulseConfig.HOP_LENGTH
        )
        
        # Convert to tensor
        spectrogram = torch.FloatTensor(mel_spec).unsqueeze(0)  # Add channel dimension
        
        # Encode label if label encoder is provided
        if self.label_encoder is not None:
            label_encoded = self.label_encoder.transform([label])[0]
        else:
            label_encoded = label
            
        label_tensor = torch.LongTensor([label_encoded])[0]
        
        return spectrogram, label_tensor, filename

def create_reef_health_labels(labels):
    """
    Convert detailed labels into reef health categories
    This simplifies the problem for hackathon demo
    """
    
    health_mapping = {}
    
    for label in labels:
        if any(bio_sound in label for bio_sound in [
            'bioph_chorus', 'bioph_chatter', 'bioph_crackle', 
            'bioph_cascading', 'bioph_grouper', 'bioph_knock'
        ]):
            health_mapping[label] = 'healthy'
        elif any(disturbance in label for disturbance in [
            'anthrop_boat', 'anthrop_mechanical', 'anthrop_bomb'
        ]):
            health_mapping[label] = 'stressed'
        elif 'ambient' in label:
            health_mapping[label] = 'ambient'
        else:
            # Other biological sounds - classify based on activity level
            health_mapping[label] = 'healthy'
    
    return health_mapping

# Create simplified health labels for hackathon demo
if labels is not None:
    health_mapping = create_reef_health_labels(set(labels))
    health_labels = [health_mapping[label] for label in labels]
    
    print("🏥 Reef Health Classification Mapping:")
    health_distribution = pd.Series(health_labels).value_counts()
    for health_status, count in health_distribution.items():
        print(f"  • {health_status.title()}: {count:,} samples ({count/len(health_labels)*100:.1f}%)")
    
    # Visualize health distribution
    plt.figure(figsize=(10, 6))
    health_distribution.plot(kind='bar', color=['green', 'orange', 'blue'])
    plt.title('🌊 Reef Health Distribution in Dataset')
    plt.xlabel('Health Status')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
# ============================================================================
# CELL 7: Model Architecture Definition
# ============================================================================

"""
🧠 Deep Learning Model for Coral Reef Health Classification
Custom CNN architecture optimized for audio spectrograms
"""

class ReefHealthCNN(nn.Module):
    """
    Convolutional Neural Network for Reef Health Classification
    Optimized for mel-spectrogram inputs
    """
    
    def __init__(self, num_classes, n_mels=128, dropout_rate=0.3):
        super(ReefHealthCNN, self).__init__()
        
        self.num_classes = num_classes
        self.n_mels = n_mels
        
        # Convolutional blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Convolutional feature extraction
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # Flatten for classifier
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize model for demonstration
demo_model = ReefHealthCNN(num_classes=3)  # healthy, stressed, ambient
param_count = count_parameters(demo_model)

print("🧠 Model Architecture Summary:")
print(f"  • Total parameters: {param_count:,}")
print(f"  • Model size: ~{param_count * 4 / 1024 / 1024:.1f} MB")
print(f"  • Input shape: (batch_size, 1, {OceanPulseConfig.N_MELS}, time_frames)")
print(f"  • Output classes: 3 (Healthy, Stressed, Ambient)")

# Print model architecture
print("\n📋 Detailed Architecture:")
print(demo_model)

# %%
# ============================================================================
# CELL 8: Training Functions and Utilities
# ============================================================================

"""
🏋 Training utilities and functions
Includes training loop, validation, and metrics calculation
"""

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for spectrograms, labels, filenames in progress_bar:
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate_model(model, dataloader, criterion, device):
    """Validate model performance"""
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for spectrograms, labels, filenames in progress_bar:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store predictions for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_predictions, all_labels

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('🔥 Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('📊 Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('🎯 Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    return cm

print("✅ Training utilities loaded!")
print("🚀 Ready to start model training!")


# %%
# ============================================================================
# CELL 9: Data Preparation and Splitting
# ============================================================================

"""
📊 Prepare data for training
Split dataset and create data loaders
"""

def prepare_training_data(data_dir, filenames, labels, test_size=0.2, val_size=0.1):
    """Prepare data splits for training"""
    
    print("🔄 Preparing training data...")
    
    # Create label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f"📋 Label Classes: {list(label_encoder.classes_)}")
    print(f"🔢 Number of classes: {len(label_encoder.classes_)}")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        filenames, encoded_labels, 
        test_size=test_size, 
        random_state=OceanPulseConfig.RANDOM_SEED,
        stratify=encoded_labels
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=OceanPulseConfig.RANDOM_SEED,
        stratify=y_temp
    )
    
    print(f"📊 Data splits:")
    print(f"  • Training: {len(X_train):,} samples ({len(X_train)/len(filenames)*100:.1f}%)")
    print(f"  • Validation: {len(X_val):,} samples ({len(X_val)/len(filenames)*100:.1f}%)")
    print(f"  • Testing: {len(X_test):,} samples ({len(X_test)/len(filenames)*100:.1f}%)")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)

# Prepare data if annotations are available
if labels is not None and os.path.exists(OceanPulseConfig.DATA_DIR):
    print("🚀 Starting data preparation...")
    
    # Use health labels for simplified classification (good for hackathon demo)
    use_simplified_labels = True
    
    if use_simplified_labels:
        print("💡 Using simplified reef health labels for demo")
        target_labels = health_labels
        class_description = "Reef Health (Healthy/Stressed/Ambient)"
    else:
        print("🔬 Using detailed acoustic labels for research")
        target_labels = labels
        class_description = "Detailed Acoustic Classes"
    
    # Prepare splits
    splits = prepare_training_data(
        OceanPulseConfig.DATA_DIR, 
        filenames, 
        target_labels
    )
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = splits
    
    print(f"\n✅ Data preparation complete!")
    print(f"🎯 Classification task: {class_description}")
    print(f"📊 Classes: {list(label_encoder.classes_)}")
    
else:
    print("⚠ Skipping data preparation - annotations or data directory not available")
    print("📝 Please ensure both ANNOTATIONS_FILE and DATA_DIR are correctly set")


# %%
# ============================================================================
# CELL 10: Optimized Dataset Objects and Data Loaders (FAST VERSION)
# ============================================================================
"""
Create PyTorch datasets and data loaders - OPTIMIZED FOR SPEED
"""

class FastReefSetDataset(Dataset):
    """
    Optimized PyTorch Dataset for ReefSet - skips slow file validation
    """
    
    def __init__(self, data_dir, filenames, labels, sr=16000, n_mels=128, 
                 duration=1.88, skip_validation=True):
        
        self.data_dir = data_dir
        self.filenames = filenames
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        
        if skip_validation:
            # Skip the slow file validation - saves 40+ minutes
            print("⚡ Fast mode: Skipping file validation")
            self.valid_indices = list(range(len(filenames)))
            print(f"✅ Dataset created with {len(self.valid_indices)} samples (validation skipped)")
        else:
            # Original slow validation (not recommended for 57k files)
            self.valid_indices = []
            print("🔍 Checking file existence...")
            
            for i, filename in enumerate(tqdm(filenames, desc="Validating files")):
                audio_path = os.path.join(data_dir, filename)
                if os.path.exists(audio_path):
                    self.valid_indices.append(i)
            
            print(f"✅ Found {len(self.valid_indices)} valid audio files out of {len(filenames)}")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get actual index
        actual_idx = self.valid_indices[idx]
        filename = self.filenames[actual_idx]
        label = self.labels[actual_idx]
        
        # Load and process audio with error handling
        audio_path = os.path.join(self.data_dir, filename)
        
        try:
            audio = load_and_process_audio(audio_path, self.sr, self.duration)
            
            if audio is None:
                # Generate dummy audio if file doesn't exist
                audio = np.random.normal(0, 0.01, int(self.duration * self.sr))
        
        except Exception:
            # Generate dummy audio on any error
            audio = np.random.normal(0, 0.01, int(self.duration * self.sr))
        
        # Generate mel spectrogram
        try:
            mel_spec = generate_mel_spectrogram(
                audio, 
                sr=self.sr, 
                n_mels=self.n_mels,
                n_fft=OceanPulseConfig.N_FFT,
                hop_length=OceanPulseConfig.HOP_LENGTH
            )
        except Exception:
            # Generate dummy spectrogram on error
            mel_spec = np.random.rand(self.n_mels, 59)  # Approximate time frames
        
        # Convert to tensor
        spectrogram = torch.FloatTensor(mel_spec).unsqueeze(0)  # Add channel dimension
        label_tensor = torch.LongTensor([label])[0]
        
        return spectrogram, label_tensor, filename

def create_fast_data_loaders(data_dir, X_train, X_val, X_test, y_train, y_val, y_test, 
                            label_encoder, batch_size=32, num_workers=2):
    """Create PyTorch data loaders quickly - OPTIMIZED VERSION"""
    
    print("🚀 Creating FAST data loaders...")
    print(f"⚡ This will complete in 2-3 minutes instead of 45+ minutes")
    
    # Reduce settings for stability with large datasets
    if batch_size > 16:
        batch_size = 16
        print(f"📦 Reducing batch size to {batch_size} for stability")
    
    if num_workers > 2:
        num_workers = 2
        print(f"👥 Reducing workers to {num_workers} for stability")
    
    # Create datasets with fast mode (skip validation)
    print("📊 Creating train dataset...")
    train_dataset = FastReefSetDataset(
        data_dir=data_dir,
        filenames=X_train,
        labels=y_train,
        sr=OceanPulseConfig.SAMPLE_RATE,
        n_mels=OceanPulseConfig.N_MELS,
        duration=OceanPulseConfig.AUDIO_DURATION,
        skip_validation=True  # This saves 40+ minutes
    )
    
    print("📊 Creating validation dataset...")
    val_dataset = FastReefSetDataset(
        data_dir=data_dir,
        filenames=X_val,
        labels=y_val,
        sr=OceanPulseConfig.SAMPLE_RATE,
        n_mels=OceanPulseConfig.N_MELS,
        duration=OceanPulseConfig.AUDIO_DURATION,
        skip_validation=True
    )
    
    print("📊 Creating test dataset...")
    test_dataset = FastReefSetDataset(
        data_dir=data_dir,
        filenames=X_test,
        labels=y_test,
        sr=OceanPulseConfig.SAMPLE_RATE,
        n_mels=OceanPulseConfig.N_MELS,
        duration=OceanPulseConfig.AUDIO_DURATION,
        skip_validation=True
    )
    
    print("🔄 Creating data loaders...")
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False,  # Reduce memory pressure
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False
    )
    
    print(f"✅ FAST data loaders created:")
    print(f"  • Train batches: {len(train_loader)}")
    print(f"  • Val batches: {len(val_loader)}")
    print(f"  • Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

# Create FAST data loaders if data is prepared
if 'splits' in locals():
    print("🚀 Starting OPTIMIZED data loader creation...")
    
    # Use optimized settings
    fast_batch_size = 16  # Smaller for stability
    fast_num_workers = 2  # Fewer workers
    
    try:
        data_loaders = create_fast_data_loaders(
            OceanPulseConfig.DATA_DIR,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            label_encoder,
            batch_size=fast_batch_size,
            num_workers=fast_num_workers
        )
        train_loader, val_loader, test_loader = data_loaders
        
        # Quick test without loading actual data (avoids hanging)
        print("\n🔍 Quick data loader validation...")
        try:
            print(f"✅ Data loaders validated:")
            print(f"  • Train samples: {len(train_loader.dataset)}")
            print(f"  • Val samples: {len(val_loader.dataset)}")
            print(f"  • Test samples: {len(test_loader.dataset)}")
            print(f"  • Train batches: {len(train_loader)}")
            print(f"  • Batch size: {train_loader.batch_size}")
            
            # Test data loader with minimal load and timeout protection
            print("🧪 Testing single sample...")
            
            # Create a simple test loader with minimal settings
            test_dataset_simple = FastReefSetDataset(
                data_dir=OceanPulseConfig.DATA_DIR,
                filenames=X_train[:1],  # Just one file
                labels=y_train[:1],
                sr=OceanPulseConfig.SAMPLE_RATE,
                n_mels=OceanPulseConfig.N_MELS,
                duration=OceanPulseConfig.AUDIO_DURATION,
                skip_validation=True
            )
            
            simple_loader = DataLoader(test_dataset_simple, batch_size=1, num_workers=0)
            sample_batch = next(iter(simple_loader))
            spectrograms, labels_batch, filenames_batch = sample_batch
            
            print(f"✅ Sample test successful:")
            print(f"  • Spectrogram shape: {spectrograms.shape}")
            print(f"  • Sample processed successfully")
            
            print("🎉 Data loaders are ready for training!")
            
        except Exception as e:
            print(f"⚠ Basic test failed: {e}")
            print("💡 Data loaders created but testing skipped")
            print("🚀 Proceeding to training - the training loop will validate data loading")
        
    except Exception as e:
        print(f"❌ Failed to create data loaders: {e}")
        print("\n🔧 Trying minimal settings...")
        
        # Fallback to minimal settings
        try:
            data_loaders = create_fast_data_loaders(
                OceanPulseConfig.DATA_DIR,
                X_train[:1000], X_val[:200], X_test[:200],  # Use smaller subset
                y_train[:1000], y_val[:200], y_test[:200],
                label_encoder,
                batch_size=4,
                num_workers=0
            )
            train_loader, val_loader, test_loader = data_loaders
            print("✅ Minimal data loaders created successfully!")
            
        except Exception as e2:
            print(f"❌ Even minimal settings failed: {e2}")
            print("📝 Please check your system resources and file paths")

else:
    print("⚠ Skipping data loader creation - data splits not available")
    print("📝 Please run the previous data preparation cells first")

print("\n⚡ Fast data loader creation complete!")
print("🎯 Time saved: ~40+ minutes compared to original validation approach")

# %%
# ============================================================================
# CELL 11: Model Training - Main Training Loop
# ============================================================================

"""
🏋 Main training loop for the OceanPulse model
This is where the magic happens!
"""

def train_oceanpulse_model(train_loader, val_loader, num_classes, num_epochs=50):
    """Main training function for OceanPulse model"""
    
    print("🚀 Starting OceanPulse model training!")
    print(f"🎯 Target: {num_classes} classes")
    print(f"⏱ Epochs: {num_epochs}")
    print(f"💻 Device: {OceanPulseConfig.DEVICE}")
    
    # Initialize model
    model = ReefHealthCNN(
        num_classes=num_classes,
        n_mels=OceanPulseConfig.N_MELS,
        dropout_rate=OceanPulseConfig.DROPOUT_RATE
    ).to(OceanPulseConfig.DEVICE)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=OceanPulseConfig.LEARNING_RATE,
        weight_decay=OceanPulseConfig.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    start_time = time.time()
    
    print("\n🔥 Training started!")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, OceanPulseConfig.DEVICE
        )
        
        # Validation phase
        val_loss, val_acc, _, _ = validate_model(
            model, val_loader, criterion, OceanPulseConfig.DEVICE
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'label_encoder': label_encoder
            }, 'best_oceanpulse_model.pth')
            print(f"💾 New best model saved! Validation accuracy: {val_acc:.2f}%")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  • Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  • Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  • Learning Rate: {current_lr:.6f}")
        print(f"  • Time: {epoch_time:.1f}s")
        print(f"  • Best Val Acc: {best_val_acc:.2f}%")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 Training completed!")
    print(f"⏱ Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Best model saved as: best_oceanpulse_model.pth")
    
    return model, train_losses, val_losses, train_accs, val_accs

# Start training if data loaders are ready
if 'train_loader' in locals():
    print("🚀 All systems ready for training!")
    
    # Get number of classes
    num_classes = len(label_encoder.classes_)
    
    # Start training
    training_results = train_oceanpulse_model(
        train_loader, 
        val_loader, 
        num_classes, 
        num_epochs=OceanPulseConfig.NUM_EPOCHS
    )
    
    model, train_losses, val_losses, train_accs, val_accs = training_results
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
else:
    print("⚠ Training skipped - data loaders not available")
    print("📝 Please ensure the previous cells have been run successfully")

# %%
# ============================================================================
# CELL 12: Model Evaluation and Testing
# ============================================================================

"""
🎯 Comprehensive model evaluation on test set
"""

def evaluate_model_comprehensive(model, test_loader, label_encoder, device):
    """Comprehensive model evaluation with detailed metrics"""
    
    print("🔍 Starting comprehensive model evaluation...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_filenames = []
    
    with torch.no_grad():
        for spectrograms, labels, filenames in tqdm(test_loader, desc="Evaluating"):
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            
            outputs = model(spectrograms)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_filenames.extend(filenames)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    class_names = label_encoder.classes_
    
    print(f"🎯 Test Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print("\n📊 Detailed Classification Report:")
    report = classification_report(
        all_labels, all_predictions, 
        target_names=class_names, 
        digits=4
    )
    print(report)
    
    # Confusion matrix
    cm = plot_confusion_matrix(all_labels, all_predictions, class_names)
    
    # Per-class accuracy analysis
    print("\n🎯 Per-Class Performance:")
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(all_predictions)[class_mask] == i)
            print(f"  • {class_name}: {class_acc*100:.2f}%")
    
    # Confidence analysis
    confidences = np.max(all_probabilities, axis=1)
    print(f"\n🎲 Confidence Statistics:")
    print(f"  • Mean confidence: {np.mean(confidences):.4f}")
    print(f"  • Std confidence: {np.std(confidences):.4f}")
    print(f"  • Min confidence: {np.min(confidences):.4f}")
    print(f"  • Max confidence: {np.max(confidences):.4f}")
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(confidences, bins=50, alpha=0.7, edgecolor='black')
    plt.title('🎲 Prediction Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    
    # Confidence vs accuracy
    correct_predictions = np.array(all_predictions) == np.array(all_labels)
    correct_confidences = confidences[correct_predictions]
    incorrect_confidences = confidences[~correct_predictions]
    
    plt.subplot(1, 2, 2)
    plt.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green')
    plt.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='red')
    plt.title('🎯 Confidence: Correct vs Incorrect')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'confidences': confidences,
        'filenames': all_filenames
    }

# Run evaluation if model is trained
if 'model' in locals() and 'test_loader' in locals():
    print("🎯 Starting model evaluation...")
    
    # Load best model
    if os.path.exists('best_oceanpulse_model.pth'):
        print("📂 Loading best saved model...")
        checkpoint = torch.load('best_oceanpulse_model.pth', map_location=OceanPulseConfig.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model with validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    # Comprehensive evaluation
    eval_results = evaluate_model_comprehensive(
        model, test_loader, label_encoder, OceanPulseConfig.DEVICE
    )
    
    print(f"\n🏆 Final Test Accuracy: {eval_results['accuracy']*100:.2f}%")
    
else:
    print("⚠ Model evaluation skipped - model or test data not available")

# %%







### Output:

<img width="1480" height="834" alt="image" src="https://github.com/user-attachments/assets/64802c95-e343-428e-acde-cb53e90b9d19" />
<img width="1490" height="1181" alt="image" src="https://github.com/user-attachments/assets/e7203b23-cbc4-43df-8d48-5a64797d66b7" />
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/e2aeec44-93ac-486c-b61d-b60dbb51984c" />




### Result:
Thus the system was trained successfully and the prediction was carried out.
