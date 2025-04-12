# Quantum Machine Learning for SUSY Classification



This repository contains an implementation of quantum machine learning techniques for classifying Supersymmetry (SUSY) particles in high-energy physics data. The project demonstrates how quantum computing can be leveraged for complex classification tasks in particle physics.

## Overview

Supersymmetry (SUSY) is a theoretical extension of the Standard Model of particle physics that predicts the existence of partner particles for each known particle. Detecting these superpartners is a challenging classification task that requires sophisticated machine learning approaches. This project explores the use of variational quantum circuits for this classification task.

## Features

- **Enhanced Quantum Classifier** with multiple circuit architectures:
  - Basic circuit with nearest-neighbor entanglement
  - Advanced circuit with improved data encoding and extended entanglement
  - Experimental circuit with strongly entangling layers
  
- **Comprehensive Training Pipeline**:
  - Automatic feature selection using mutual information
  - Data scaling techniques optimized for quantum processing
  - Adaptive learning rate scheduling
  - Early stopping to prevent overfitting
  - Batch processing for improved training efficiency
  
- **Extensive Evaluation Metrics**:
  - ROC curves and AUC calculation
  - Precision-Recall analysis
  - Confusion matrices
  - F1 scores
  
- **Interactive Dashboard**:
  - HTML-based visualization dashboard
  - Comparative analysis across model configurations
  - Training history tracking
  - Feature importance visualization

## Requirements

- Python 3.7+
- PennyLane
- NumPy
- scikit-learn
- Matplotlib
- seaborn
- pandas
- tqdm

## Installation

```bash
# Clone the repository
git clone https://github.com/bhuvaneshwar9/quantum-ml.git
cd quantum-ml

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

To run the default experiment with multiple model configurations:

```python
python main.py
```

This will:
1. Load and preprocess the SUSY dataset
2. Train multiple quantum classifier configurations
3. Evaluate performance and generate visualizations
4. Create a comprehensive dashboard for result analysis

### Custom Experiment

You can customize the experiment by modifying the configurations in `main.py`:

```python
# Run a single experiment with custom parameters
classifier, result, selected_features, feature_scores = run_enhanced_susy_experiment(
    susy_file,
    n_qubits=4,               # Number of qubits to use
    n_layers=2,               # Depth of the quantum circuit
    train_samples=200,        # Number of training samples
    test_samples=100,         # Number of test samples
    steps=100,                # Training iterations
    circuit_type="advanced",  # Circuit architecture ("basic", "advanced", or "experimental")
    learning_rate=0.05        # Learning rate for optimization
)
```

## Dataset

The project uses the SUSY dataset from the UCI Machine Learning Repository:

- **Dataset**: [SUSY Dataset](https://archive.ics.uci.edu/ml/datasets/SUSY)
- **Features**: 18 kinematic properties measured by particle detectors
- **Target**: Binary classification (signal vs background)
- **Size**: 5 million examples (a subset is used for quantum processing)

Place the `SUSY.csv` file in the project root directory.

## Results

The dashboard provides a comprehensive view of the model performance:

- **Model Comparison**: Different quantum circuit architectures are compared based on accuracy, F1 score, ROC AUC, and PR AUC
- **ROC Analysis**: Visualizes the trade-off between true positive rate and false positive rate
- **Precision-Recall**: Shows the trade-off between precision and recall at different thresholds
- **Confusion Matrices**: Visual representation of prediction accuracy
- **Feature Importance**: Identifies the most relevant features for classification

## Project Structure

```
quantum-ml/
├── main.py                    # Main script with quantum classifier implementation
├── quantum_results/           # Generated results directory
│   ├── plots/                 # Visualization plots
│   ├── models/                # Saved model weights
│   └── quantum_dashboard.html # Interactive results dashboard
├── SUSY.csv                   # Dataset (to be downloaded separately)
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## How It Works

1. **Data Preprocessing**:
   - Feature selection to identify the most relevant attributes
   - Scaling to prepare data for quantum processing
   - Train/validation/test splitting for proper evaluation

2. **Quantu
