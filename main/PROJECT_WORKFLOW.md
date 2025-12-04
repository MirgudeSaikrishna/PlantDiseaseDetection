# 🌿 Plant Disease Detection System - Complete Workflow Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Model Architecture](#model-architecture)
4. [Model Training](#model-training)
5. [Model Testing](#model-testing)
6. [Model Storage](#model-storage)
7. [Application Deployment](#application-deployment)
8. [File Structure](#file-structure)
9. [Data Flow](#data-flow)

---

## 🎯 Project Overview

**Project Name:** AgriDetect AI - Plant Disease Detection System  
**Purpose:** Detect 39 different plant diseases using deep learning CNN models  
**Technology Stack:** PyTorch, Flask, React, SQLite  
**Location:** `main/app/`

---

## 📊 1. Dataset Preparation

### Dataset Location
- **Expected Location:** `Dataset/` folder (not in repository, needs to be provided)(https://data.mendeley.com/datasets/tywbtsjrjv/1)
- **Structure:** ImageFolder format with class-based subdirectories
- **Format:** 
  ```
  Dataset/
  ├── Apple___Apple_scab/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── Apple___Black_rot/
  ├── Apple___Cedar_apple_rust/
  └── ... (39 classes total)
  ```

### Dataset Information
- **Total Classes:** 39 plant disease categories
- **Classes Include:**
  - Apple diseases (Scab, Black Rot, Cedar Apple Rust, Healthy)
  - Corn diseases (Cercospora, Common Rust, Northern Leaf Blight, Healthy)
  - Tomato diseases (10+ types including Bacterial Spot, Blights, Viruses)
  - Grape, Potato, Pepper, Cherry, and more

### Dataset Processing
**File:** `main/app/train_improved_model.py` (Lines 39-78)

**Process:**
1. **Data Transforms** (Lines 39-58):
   - **Training Transforms:**
     - Resize to 256x256
     - Random crop to 224x224
     - Random horizontal flip (50% probability)
     - Random rotation (±15 degrees)
     - Color jitter (brightness, contrast, saturation, hue)
     - Random affine transformations
     - Normalize with ImageNet statistics
   
   - **Validation/Test Transforms:**
     - Resize to 224x224
     - Convert to tensor
     - Normalize with ImageNet statistics

2. **Dataset Splitting** (Lines 60-78):
   - **70% Training** - Used for model learning
   - **15% Validation** - Used for hyperparameter tuning and early stopping
   - **15% Test** - Used for final evaluation

3. **Data Loaders** (Lines 80-86):
   - Batch size: 32 (configurable)
   - Shuffle: True for training, False for validation/test
   - Number of workers: 4 (for parallel data loading)

---

## 🏗️ 2. Model Architecture

### Model Files
- **Location:** `main/app/CNN.py`
- **Two Models Available:**

#### A. Original CNN Model (Lines 100-158)
**Class:** `CNN`
- **Architecture:**
  - 4 Convolutional Blocks
  - Each block: Conv2d → ReLU → BatchNorm → Conv2d → ReLU → BatchNorm → MaxPool2d
  - Channels: 3 → 32 → 64 → 128 → 256
  - Dense layers: 50176 → 1024 → 39 (output classes)
  - Dropout: 0.4

#### B. Improved CNN Model (Lines 5-97) ⭐ **Currently Used**
**Class:** `ImprovedCNN`
- **Enhanced Features:**
  - 5 Convolutional Blocks with residual connections
  - Global Average Pooling (reduces overfitting)
  - Attention mechanism for feature focus
  - Advanced regularization (dropout + batch normalization)
  - Xavier weight initialization
  - Channels: 3 → 32 → 64 → 128 → 256 → 512

**Architecture Flow:**
```
Input (3, 224, 224)
  ↓
Conv Block 1 (32 channels)
  ↓
Conv Block 2 (64 channels)
  ↓
Conv Block 3 (128 channels)
  ↓
Conv Block 4 (256 channels)
  ↓
Conv Block 5 (512 channels)
  ↓
Global Average Pooling (512)
  ↓
Attention Mechanism
  ↓
Classifier: 512 → 1024 → 512 → 39
  ↓
Output (39 classes)
```

### Class Mapping
**File:** `main/app/CNN.py` (Lines 161-199)
- Dictionary mapping: `idx_to_classes`
- Maps class indices (0-38) to disease names
- Example: `0: 'Apple___Apple_scab'`, `38: 'Tomato___healthy'`

---

## 🎓 3. Model Training

### Training Script
**File:** `main/app/train_improved_model.py`

### Training Process (Step-by-Step)

#### Step 1: Initialization (Lines 28-37)
```python
trainer = PlantDiseaseTrainer(
    model_name="improved_cnn",
    num_classes=39,
    device='cuda' if available else 'cpu'
)
```

#### Step 2: Create Data Transforms (Lines 39-58)
- Training transforms with augmentation
- Validation transforms without augmentation

#### Step 3: Load Dataset (Lines 60-78)
- Load from `Dataset/` folder
- Split into train/val/test (70/15/15)

#### Step 4: Create Data Loaders (Lines 80-86)
- Batch size: 32
- Parallel loading with 4 workers

#### Step 5: Initialize Model (Lines 88-96)
- Create ImprovedCNN with 39 classes
- Move to GPU if available

#### Step 6: Training Loop (Lines 150-209)
**Configuration:**
- **Epochs:** 50 (configurable)
- **Learning Rate:** 0.001
- **Optimizer:** Adam with weight decay (1e-4)
- **Loss Function:** CrossEntropyLoss
- **Scheduler:** ReduceLROnPlateau (reduces LR when validation loss plateaus)
- **Early Stopping:** Patience = 10 epochs

**Training Process:**
1. **For each epoch:**
   - Set model to training mode
   - For each batch:
     - Forward pass
     - Calculate loss
     - Backward pass (compute gradients)
     - Update weights (optimizer step)
     - Track accuracy
   
2. **Validation:**
   - Set model to evaluation mode
   - No gradient computation
   - Calculate validation loss and accuracy
   - Update learning rate scheduler

3. **Early Stopping:**
   - Monitor validation loss
   - If no improvement for 10 epochs → stop training
   - Save best model state

4. **Metrics Tracking:**
   - Train loss per epoch
   - Validation loss per epoch
   - Train accuracy per epoch
   - Validation accuracy per epoch

#### Step 7: Model Evaluation (Lines 211-231)
- Test on held-out test set
- Calculate test accuracy
- Generate predictions and targets for analysis

#### Step 8: Save Results (Lines 259-281)
- **Model:** Saved as `.pt` file (PyTorch state dict)
- **Training Info:** Saved as JSON with metrics
- **Plots:** Training history visualization

### Training Output Files
**Location:** `main/app/`
- `improved_plant_disease_model.pt` - Trained model weights
- `training_info.json` - Training metadata and metrics
- `training_history.png` - Loss and accuracy plots

---

## 🧪 4. Model Testing

### Test Files Location
**Directory:** `main/app/tests/`

### Test Files:

#### A. Model Tests (`test_model.py`)
**Tests:**
1. **CNN Initialization** - Verify model structure
2. **Forward Pass** - Test input/output shapes
3. **Parameter Count** - Verify model size
4. **Improved CNN Features:**
   - Attention mechanism
   - Dropout behavior
   - Parameter comparison
5. **Robustness Tests:**
   - Different input sizes
   - Edge cases (zeros, ones, noise)
   - Memory usage

#### B. API Tests (`test_api.py`)
**Tests:**
1. **Detection Endpoint** (`/api/detect`)
   - Successful detection
   - Missing image handling
   - Location parameter
2. **Batch Detection** (`/api/batch-detect`)
   - Multiple images
   - Limit enforcement (max 10)
3. **Analytics Endpoints:**
   - Health report (`/api/health-report`)
   - Detection history (`/api/history`)
4. **Legacy Endpoints:**
   - Home, contact, market pages
   - Submit endpoint

#### C. Advanced Features Tests (`test_advanced_features.py`)
- Tests for severity scoring
- Weather integration
- Treatment plan generation
- Historical tracking

### Running Tests
**File:** `main/app/run_tests.py`
```bash
cd main/app
python run_tests.py
# OR
pytest tests/
```

---

## 💾 5. Model Storage

### Model Files

#### A. Trained Model
**File:** `main/app/plant_disease_model_1_latest.pt`
- **Format:** PyTorch state dictionary (`.pt`)
- **Content:** Trained weights and biases
- **Size:** ~50-100 MB (approximate)
- **Usage:** Loaded in `app.py` for inference

#### B. Model Loading Process
**File:** `main/app/app.py` (Lines 24-27)
```python
model = CNN.CNN(39)  # Initialize model architecture
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()  # Set to evaluation mode
```

### Data Storage

#### A. Detection History Database
**File:** `main/app/plant_detection_history.db`
- **Type:** SQLite database
- **Schema:** `detection_history` table
- **Fields:**
  - `id` (Primary Key)
  - `timestamp` (DateTime)
  - `image_path` (Text)
  - `disease_name` (Text)
  - `confidence` (Real)
  - `severity_score` (Real)
  - `location` (Text)
  - `weather_condition` (Text)
  - `treatment_applied` (Text)
  - `notes` (Text)

**Initialization:** `main/app/advanced_features.py` (Lines 38-59)

#### B. CSV Data Files
1. **Disease Information**
   - **File:** `main/app/disease_info.csv`
   - **Columns:** index, disease_name, description, Possible Steps, image_url
   - **Usage:** Provides disease descriptions and prevention steps

2. **Supplement Information**
   - **File:** `main/app/supplement_info.csv`
   - **Columns:** index, disease_name, supplement name, supplement image, buy link
   - **Usage:** Links diseases to treatment products

#### C. Uploaded Images
**Directory:** `main/app/static/uploads/`
- Stores user-uploaded images
- Filename format: `YYYYMMDD_HHMMSS_originalname.jpg`
- Used for predictions and history tracking

---

## 🚀 6. Application Deployment

### Main Application File
**File:** `main/app/app.py`

### Application Flow

#### A. Initialization (Lines 16-31)
1. **Flask App Setup:**
   - Initialize Flask application
   - Enable CORS for API access
   
2. **Data Loading:**
   - Load `disease_info.csv`
   - Load `supplement_info.csv`
   
3. **Model Loading:**
   - Initialize CNN model (39 classes)
   - Load pre-trained weights
   - Set to evaluation mode
   
4. **Advanced Analyzer:**
   - Initialize `AdvancedPlantAnalyzer`
   - Set up database connection

#### B. Prediction Functions

**1. Basic Prediction** (Lines 33-42)
- **Function:** `prediction(image_path)`
- **Process:**
  - Load image
  - Resize to 224x224
  - Convert to tensor
  - Forward pass through model
  - Return class index

**2. Advanced Prediction** (Lines 44-60)
- **Function:** `advanced_prediction(image_path, location)`
- **Features:**
  - Disease detection
  - Confidence scoring
  - Severity assessment
  - Health score calculation
  - Treatment plan generation
  - Weather integration (if location provided)
  - Database storage

#### C. API Endpoints

**1. Disease Detection** (`/api/detect`) - Lines 64-105
- **Method:** POST
- **Input:** Image file + optional location
- **Output:** JSON with disease info, confidence, severity, treatment plan
- **Process:**
  1. Receive image
  2. Save to `static/uploads/`
  3. Run advanced prediction
  4. Match with CSV data
  5. Return comprehensive result

**2. Batch Detection** (`/api/batch-detect`) - Lines 107-132
- **Method:** POST
- **Input:** Multiple images (max 10)
- **Output:** Array of results
- **Process:** Process each image sequentially

**3. Health Report** (`/api/health-report`) - Lines 134-143
- **Method:** GET
- **Parameters:** `days` (default: 30)
- **Output:** Statistical report of detections

**4. Detection History** (`/api/history`) - Lines 145-154
- **Method:** GET
- **Parameters:** `days` (default: 30)
- **Output:** List of past detections

#### D. Web Routes (Legacy Support)

1. **Home Page** (`/`) - Line 157
2. **Contact Page** (`/contact`) - Line 161
3. **AI Engine** (`/index`) - Line 165
4. **Mobile Device** (`/mobile-device`) - Line 169
5. **Submit** (`/submit`) - Lines 173-190
   - Handles image upload
   - Runs prediction
   - Renders results page
6. **Market** (`/market`) - Lines 192-195
   - Displays supplement marketplace

### Advanced Features Module
**File:** `main/app/advanced_features.py`

#### Key Classes:

**1. AdvancedPlantAnalyzer** (Lines 28-389)
- **Methods:**
  - `analyze_plant_health()` - Comprehensive analysis
  - `batch_analyze()` - Process multiple images
  - `_calculate_severity_score()` - Visual severity analysis
  - `_calculate_health_score()` - Overall health metric
  - `_get_weather_data()` - Weather API integration
  - `_generate_treatment_plan()` - Treatment recommendations
  - `_store_detection()` - Save to database
  - `get_detection_history()` - Retrieve past detections
  - `generate_health_report()` - Analytics report

**2. MobileOptimizer** (Lines 391-423)
- Image compression for mobile
- Mobile-optimized response format

### Utility Functions
**File:** `main/app/utils.py`
- Image preprocessing
- File validation
- Logging setup
- Rate limiting
- Error handling

### Configuration
**File:** `main/app/config.py`
- Development, Testing, Production configs
- Environment variables
- Security settings

---

## 📁 7. File Structure

```
main/
├── app/
│   ├── __pycache__/          # Python bytecode
│   ├── static/
│   │   └── uploads/           # User uploaded images
│   ├── templates/            # HTML templates
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── index.html
│   │   ├── submit.html
│   │   ├── market.html
│   │   └── contact-us.html
│   ├── tests/                # Test files
│   │   ├── __init__.py
│   │   ├── test_model.py
│   │   ├── test_api.py
│   │   └── test_advanced_features.py
│   ├── venv/                 # Virtual environment
│   ├── CNN.py                # Model architecture
│   ├── app.py                # Main Flask application
│   ├── train_improved_model.py  # Training script
│   ├── advanced_features.py    # Advanced features
│   ├── utils.py              # Utility functions
│   ├── config.py             # Configuration
│   ├── run_tests.py          # Test runner
│   ├── disease_info.csv      # Disease information
│   ├── supplement_info.csv  # Supplement information
│   ├── plant_disease_model_1_latest.pt  # Trained model
│   ├── plant_detection_history.db       # SQLite database
│   ├── requirements.txt      # Python dependencies
│   ├── pytest.ini           # Pytest configuration
│   ├── Procfile             # Heroku deployment
│   └── Readme.md            # App documentation
├── demo_images/             # Demo images
├── test_images/             # Test images
├── Model/                   # Model documentation
│   ├── Plant Disease Detection Code.ipynb
│   └── model.JPG
└── README.md                # Main project README
```

---

## 🔄 8. Data Flow

### Training Flow
```
Dataset/
  ↓
Data Transforms (Augmentation)
  ↓
Data Loaders (Batches)
  ↓
CNN Model (Forward Pass)
  ↓
Loss Calculation
  ↓
Backward Pass (Gradients)
  ↓
Optimizer Update
  ↓
Validation
  ↓
Early Stopping Check
  ↓
Save Best Model
```

### Inference Flow
```
User Uploads Image
  ↓
Save to static/uploads/
  ↓
Preprocess Image (Resize, Normalize)
  ↓
CNN Model (Forward Pass)
  ↓
Get Predictions (Class + Confidence)
  ↓
Advanced Analysis:
  - Severity Score
  - Health Score
  - Treatment Plan
  - Weather Data (if location provided)
  ↓
Match with CSV Data:
  - Disease Description
  - Prevention Steps
  - Supplement Info
  ↓
Store in Database
  ↓
Return JSON Response
```

### Database Flow
```
Detection Request
  ↓
Analyze Image
  ↓
Store Detection:
  - Timestamp
  - Image Path
  - Disease Name
  - Confidence
  - Severity
  - Location
  - Weather
  ↓
SQLite Database
  ↓
Retrieve for Reports/History
```

---

## 🔑 Key Configuration Points

### Model Configuration
- **Input Size:** 224x224 pixels
- **Number of Classes:** 39
- **Model Type:** ImprovedCNN (with attention)
- **Confidence Threshold:** 0.7 (configurable)

### Training Configuration
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Epochs:** 50 (with early stopping)
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss

### Application Configuration
- **Max File Size:** 16 MB
- **Allowed Formats:** PNG, JPG, JPEG, GIF, BMP, WEBP
- **Batch Limit:** 10 images per batch
- **Database:** SQLite (plant_detection_history.db)

---

## 📝 Important Notes

1. **Dataset:** The actual training dataset is not included in the repository. You need to provide it in the `Dataset/` folder with ImageFolder structure.

2. **Model File:** The pre-trained model `plant_disease_model_1_latest.pt` must be present in `main/app/` for the application to run.

3. **Dependencies:** Install all requirements from `requirements.txt` before running.

4. **Database:** SQLite database is created automatically on first run.

5. **Weather API:** Requires OpenWeatherMap API key (optional, for weather integration).

6. **GPU Support:** Training automatically uses GPU if available, otherwise falls back to CPU.

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Author:** Project Team

