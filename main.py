import numpy as np
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, precision_recall_curve, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import os
from tqdm import tqdm
import pandas as pd

    # Set a consistent visual style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Blues")
PLOT_STYLE = {
    'figsize': (12, 9),
    'titlesize': 18,
    'labelsize': 14,
    'legendsize': 12,
    'linewidth': 2.5,
    'markersize': 8,
    'dpi': 300
}

# Helper function to track progress with better formatting
def print_progress(message, level=0):
    """Print progress with timestamps and optional indentation levels"""
    prefix = "  " * level
    time_str = time.strftime('%H:%M:%S')
    print(f"[{time_str}] {prefix}⚛️ {message}")
    sys.stdout.flush()

def create_directory(dir_name):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print_progress(f"Created directory: {dir_name}")
    return dir_name

# Create output directories
results_dir = create_directory("quantum_results")
plots_dir = create_directory(os.path.join(results_dir, "plots"))
model_dir = create_directory(os.path.join(results_dir, "models"))

class EnhancedQuantumClassifier:
    """
    Enhanced Quantum Classifier with improved circuit architecture,
    training methods, and visualization capabilities
    """
    
    def __init__(self, n_qubits=4, n_layers=2, learning_rate=0.05, shots=1000, seed=42,
                 circuit_type="advanced", activation="relu", name="QuantumModel"):
        """Initialize the quantum classifier with enhanced parameters"""
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.shots = shots
        self.seed = seed
        self.circuit_type = circuit_type
        self.activation = activation
        self.name = name
        
        print_progress(f"Initializing {self.name} with {n_qubits} qubits, {n_layers} layers, circuit: {circuit_type}")
        
        # Set seeds for reproducibility
        np.random.seed(self.seed)
        
        # Initialize weights with improved initialization
        scale = 0.01  # Smaller initial weights for better stability
        self.weights = scale * np.random.randn(n_layers, n_qubits, 3)
        
        # Create quantum device with sufficient shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        
        # Select and define the quantum circuit based on type
        if circuit_type == "basic":
            self.circuit = self._create_basic_circuit()
        elif circuit_type == "advanced":
            self.circuit = self._create_advanced_circuit()
        else:
            self.circuit = self._create_experimental_circuit()
            
        # Training history
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'best_step': 0,
            'best_loss': float('inf')
        }
    
    def _create_basic_circuit(self):
        """Create a basic quantum circuit similar to the original"""
        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            # Encode inputs
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Apply variational layers
            for layer in range(self.n_layers):
                # Rotation gates
                for q in range(self.n_qubits):
                    qml.RX(weights[layer, q, 0], wires=q)
                    qml.RY(weights[layer, q, 1], wires=q)
                    qml.RZ(weights[layer, q, 2], wires=q)
                
                # Entangling gates - nearest-neighbor
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                
                # Add long-range entanglement
                if self.n_qubits > 2:
                    qml.CNOT(wires=[0, self.n_qubits - 1])
            
            # Measure Z expectation on first qubit
            return qml.expval(qml.PauliZ(0))
        
        return circuit
    
    def _create_advanced_circuit(self):
        """Create an advanced quantum circuit with improved architecture"""
        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            # Improved data encoding - amplitude encoding
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
                qml.RY(inputs[i] * np.pi, wires=i)  # Additional rotation for better encoding
            
            # Apply variational layers with improved structure
            for layer in range(self.n_layers):
                # Full rotation gates
                for q in range(self.n_qubits):
                    qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
                
                # Entangling structure - nearest-neighbor for stability
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                
                # Add CZ gates for expressivity (in a way compatible with PennyLane)
                if layer % 2 == 0 and self.n_qubits > 2:  # Alternate layer structures
                    qml.CZ(wires=[0, self.n_qubits - 1])
            
            # Measure Z expectation on first qubit
            return qml.expval(qml.PauliZ(0))
        
        return circuit
    
    def _create_experimental_circuit(self):
        """Create an experimental circuit with advanced features"""
        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            # Standard embedding for stability
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Apply strongly entangling layers
            for layer in range(self.n_layers):
                # Rotation gates
                for q in range(self.n_qubits):
                    qml.RX(weights[layer, q, 0], wires=q)
                    qml.RY(weights[layer, q, 1], wires=q)
                    qml.RZ(weights[layer, q, 2], wires=q)
                
                # Entangling gates
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                
                # Long-range entanglement
                if self.n_qubits > 2:
                    qml.CNOT(wires=[0, self.n_qubits - 1])
            
            # Measure Z expectation on first qubit
            return qml.expval(qml.PauliZ(0))
        
        return circuit
    
    def _activation_function(self, x):
        """Apply non-linear activation to raw circuit output"""
        if self.activation == "relu":
            return max(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        else:  # linear or none
            return x
    
    def predict_proba(self, X):
        """Get probability predictions for samples with improved handling"""
        n_samples = len(X)
        probas = np.zeros((n_samples, 2))
        
        for i, x in enumerate(X):
            # Ensure input is properly formatted
            if len(x) < self.n_qubits:
                x_pad = np.pad(x, (0, self.n_qubits - len(x)))
            else:
                x_pad = x[:self.n_qubits]
            
            # Get raw prediction from quantum circuit
            z_exp = self.circuit(x_pad, self.weights)
            
            # Apply activation function for better probability mapping
            processed_output = self._activation_function(z_exp)
            
            # Convert to probability with balanced distribution
            prob = (processed_output + 1) / 2  # Map from [-1,1] to [0,1]
            prob = min(max(prob, 0.001), 0.999)  # Avoid extreme values
            
            # Store probabilities for both classes
            probas[i] = [1 - prob, prob]
        
        return probas
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions with customizable threshold"""
        probas = self.predict_proba(X)
        return (probas[:, 1] > threshold).astype(int)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, steps=100, batch_size=16, 
            validation_interval=5, early_stopping_patience=10, verbose=True):
        """Train the quantum classifier with improved methodology"""
        if verbose:
            print_progress(f"Training {self.name} for {steps} steps with batch size {batch_size}")
        
        # Define cost function with L2 regularization
        def cost_function(weights, X_batch, y_batch, reg_lambda=0.01):
            # Get predictions
            preds = []
            for x in X_batch:
                # Pad/truncate input
                if len(x) < self.n_qubits:
                    x_pad = np.pad(x, (0, self.n_qubits - len(x)))
                else:
                    x_pad = x[:self.n_qubits]
                
                # Get quantum prediction
                pred = self.circuit(x_pad, weights)
                preds.append(pred)
            
            # Convert predictions and targets for loss calculation
            preds = np.array(preds)
            targets = 2 * y_batch - 1  # Convert from {0,1} to {-1,1}
            
            # Mean squared error loss with L2 regularization
            loss = np.mean((preds - targets) ** 2)
            
            # Add L2 regularization term
            reg_term = reg_lambda * np.sum(weights**2)
            
            return loss + reg_term
        
        # Initialize optimizer with learning rate scheduling
        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        
        # Initialize early stopping tracking
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = self.weights.copy()
        
        # Training loop with progress bar 
        n_samples = len(X_train)
        n_batches = int(np.ceil(n_samples / batch_size))
        
        progress_bar = tqdm(total=steps, desc="Training", disable=not verbose)
        
        for step in range(steps):
            # Learning rate decay
            if step > 0 and step % 20 == 0:
                opt.stepsize *= 0.9  # Reduce learning rate gradually
            
            # Create batch indices
            if step % n_batches == 0:
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
            
            # Get batch
            batch_idx = (step % n_batches) * batch_size
            X_batch = X_shuffled[batch_idx:min(batch_idx + batch_size, n_samples)]
            y_batch = y_shuffled[batch_idx:min(batch_idx + batch_size, n_samples)]
            
            # Update weights with gradient clipping
            self.weights = opt.step(lambda w: cost_function(w, X_batch, y_batch), self.weights)
            
            # Clip weights to prevent exploding gradients
            self.weights = np.clip(self.weights, -5, 5)
            
            # Calculate and store metrics (every validation_interval steps)
            if step % validation_interval == 0 or step == steps - 1:
                # Use a subset for training metrics for efficiency
                subset_size = min(100, n_samples)
                train_loss = cost_function(self.weights, X_train[:subset_size], y_train[:subset_size], reg_lambda=0)
                self.history['loss'].append(float(train_loss))
                
                # Calculate validation metrics if validation data provided
                if X_val is not None and y_val is not None:
                    val_subset_size = min(100, len(X_val))
                    val_loss = cost_function(self.weights, X_val[:val_subset_size], y_val[:val_subset_size], reg_lambda=0)
                    self.history['val_loss'].append(float(val_loss))
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_weights = self.weights.copy()
                        self.history['best_step'] = step
                        self.history['best_loss'] = float(best_val_loss)
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        print_progress(f"Early stopping at step {step}")
                        self.weights = best_weights
                        break
                
                if verbose and step % 10 == 0:
                    val_msg = f", Val Loss: {val_loss:.4f}" if X_val is not None else ""
                    print_progress(f"Step {step}: Loss = {train_loss:.4f}{val_msg}", level=1)
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Save model weights
        self.save_model()
        
        if X_val is not None and self.weights is not best_weights:
            print_progress(f"Restoring best weights from step {self.history['best_step']}")
            self.weights = best_weights
        
        return self.history
    
    def save_model(self):
        """Save model weights to file"""
        model_path = os.path.join(model_dir, f"{self.name}_weights.npy")
        np.save(model_path, self.weights)
        print_progress(f"Model weights saved to {model_path}")
    
    def load_model(self, model_name=None):
        """Load model weights from file"""
        if model_name is None:
            model_name = self.name
        model_path = os.path.join(model_dir, f"{model_name}_weights.npy")
        if os.path.exists(model_path):
            self.weights = np.load(model_path)
            print_progress(f"Model weights loaded from {model_path}")
            return True
        else:
            print_progress(f"No saved weights found at {model_path}")
            return False
    
    def plot_training_history(self):
        """Plot training and validation loss history"""
        plt.figure(figsize=PLOT_STYLE['figsize'])
        
        # Plot training loss
        plt.plot(range(0, len(self.history['loss']) * 5, 5), 
                 self.history['loss'], 
                 label='Training Loss', 
                 linewidth=PLOT_STYLE['linewidth'])
        
        # Plot validation loss if available
        if 'val_loss' in self.history and len(self.history['val_loss']) > 0:
            plt.plot(range(0, len(self.history['val_loss']) * 5, 5), 
                     self.history['val_loss'], 
                     label='Validation Loss', 
                     linewidth=PLOT_STYLE['linewidth'])
            
            # Mark the best epoch
            best_step = self.history.get('best_step', 0)
            best_val_idx = best_step // 5
            if best_val_idx < len(self.history['val_loss']):
                plt.scatter([best_step], [self.history['val_loss'][best_val_idx]], 
                            color='green', s=150, label='Best Model', zorder=5)
        
        plt.xlabel('Training Step', fontsize=PLOT_STYLE['labelsize'])
        plt.ylabel('Loss', fontsize=PLOT_STYLE['labelsize'])
        plt.title(f'Training History for {self.name}', fontsize=PLOT_STYLE['titlesize'])
        plt.legend(fontsize=PLOT_STYLE['legendsize'])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, f"{self.name}_training_history.png")
        plt.savefig(plot_path, dpi=PLOT_STYLE['dpi'])
        print_progress(f"Training history plot saved to {plot_path}")
        
        return plt


def load_and_process_susy_data(file_path, max_samples=10000, test_size=0.2, val_size=0.1,
                               n_features=8, feature_selection_method="f_classif", 
                               random_state=42, scaling_method="minmax"):
    """Load and preprocess the SUSY dataset with enhanced feature selection"""
    print_progress(f"Loading SUSY dataset from {file_path}")
    try:
        # Load the data
        data = np.loadtxt(file_path, delimiter=',', max_rows=max_samples)
        
        # Extract features and target
        y = data[:, 0]  # First column is the class label
        X = data[:, 1:]  # Remaining columns are features
        
        print_progress(f"Loaded {len(X)} samples with {X.shape[1]} features")
        
        # Handle class imbalance information
        class_counts = np.bincount(y.astype(int))
        class_balance = class_counts / len(y)
        print_progress(f"Class distribution: {class_balance[0]:.2f}/{class_balance[1]:.2f}")
        
        # Apply feature selection based on method
        if feature_selection_method == "f_classif":
            selector = SelectKBest(f_classif, k=n_features)
        elif feature_selection_method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=n_features)
        else:
            raise ValueError(f"Unknown feature selection method: {feature_selection_method}")
        
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        feature_scores = selector.scores_
        
        # Print feature importance information
        print_progress(f"Selected {n_features} features with indices: {selected_indices}")
        print_progress(f"Feature scores: {feature_scores[selected_indices]}")
        
        # Scale features based on method
        if scaling_method == "minmax":
            scaler = MinMaxScaler(feature_range=(-np.pi/2, np.pi/2))
        elif scaling_method == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        X_scaled = scaler.fit_transform(X_selected)
        
        # Create train/val/test split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Create validation split from training data
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)  # Adjust for the remaining size
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size_adjusted, 
                random_state=random_state, stratify=y_train_val
            )
            print_progress(f"Data split: {X_train.shape[0]} train, {X_val.shape[0]} validation, {X_test.shape[0]} test samples")
            return X_train, X_val, X_test, y_train, y_val, y_test, selected_indices, feature_scores
        else:
            print_progress(f"Data split: {X_train_val.shape[0]} train, {X_test.shape[0]} test samples")
            return X_train_val, None, X_test, y_train_val, None, y_test, selected_indices, feature_scores
    
    except Exception as e:
        print_progress(f"Error loading SUSY data: {str(e)}")
        raise


def plot_feature_importance(selected_indices, feature_scores, original_features=None):
    """Plot feature importance scores"""
    plt.figure(figsize=PLOT_STYLE['figsize'])
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_scores[selected_indices])
    x = np.arange(len(selected_indices))
    
    # Create labels
    if original_features is None:
        original_features = [f"Feature {i+1}" for i in range(max(selected_indices)+1)]
    
    feature_names = [original_features[i] for i in selected_indices]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    # Create bar chart
    bars = plt.barh(x, feature_scores[selected_indices][sorted_idx], height=0.6, color='blue')
    plt.yticks(x, sorted_names, fontsize=PLOT_STYLE['labelsize'])
    plt.xlabel('Importance Score', fontsize=PLOT_STYLE['labelsize'])
    plt.title('Feature Importance for SUSY Classification', fontsize=PLOT_STYLE['titlesize'])
    
    # Add values to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1 * max(feature_scores[selected_indices]),
                 bar.get_y() + bar.get_height()/2,
                 f'{width:.1f}',
                 ha='left', va='center', fontsize=PLOT_STYLE['labelsize']-2)
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(plots_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=PLOT_STYLE['dpi'])
    print_progress(f"Feature importance plot saved to {plot_path}")
    
    return plt


def evaluate_classifier(classifier, X_test, y_test, threshold=0.5):
    """Evaluate classifier with comprehensive metrics and visualizations"""
    print_progress(f"Evaluating {classifier.name} on test set")
    
    # Make predictions
    y_probas = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test, threshold=threshold)
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_probas[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_probas[:, 1])
    pr_auc = auc(recall, precision)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print detailed results
    print_progress(f"Test accuracy: {accuracy:.4f}")
    print_progress(f"Test F1 score: {f1:.4f}")
    print_progress(f"Test ROC AUC: {roc_auc:.4f}")
    print_progress(f"Test PR AUC: {pr_auc:.4f}")
    
    # Return all metrics
    return {
        'name': classifier.name,
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'confusion_matrix': cm.tolist(),
        'y_probas': y_probas.tolist(),
        'y_pred': y_pred.tolist(),
        'y_true': y_test.tolist()
    }


def plot_roc_curve(results_dict, fig=None, ax=None):
    """Plot ROC curve with enhanced styling"""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'])
    
    for name, result in results_dict.items():
        ax.plot(result['fpr'], result['tpr'],
                label=f"{name} (AUC = {result['roc_auc']:.4f})",
                linewidth=PLOT_STYLE['linewidth'])
    
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5)
    
    # Add styling
    ax.set_xlabel('False Positive Rate', fontsize=PLOT_STYLE['labelsize'])
    ax.set_ylabel('True Positive Rate', fontsize=PLOT_STYLE['labelsize'])
    ax.set_title('ROC Curves for SUSY Classification', fontsize=PLOT_STYLE['titlesize'])
    ax.legend(loc='lower right', fontsize=PLOT_STYLE['legendsize'])
    ax.grid(alpha=0.3)
    
    # Add shaded regions for model quality indicators
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.1, color='green', label='_nolegend_')
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.1, color='red', label='_nolegend_')
    
    # Add annotations
    ax.annotate('Better', xy=(0.25, 0.75), fontsize=12, color='green')
    ax.annotate('Worse', xy=(0.75, 0.25), fontsize=12, color='red')
    
    # Save the figure
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "roc_comparison.png")
    plt.savefig(plot_path, dpi=PLOT_STYLE['dpi'])
    print_progress(f"ROC comparison plot saved to {plot_path}")
    
    return fig, ax


def plot_precision_recall_curve(results_dict):
    """Plot precision-recall curve with enhanced styling"""
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'])
    
    # Calculate and plot F1 isolines
    f1_levels = [0.2, 0.4, 0.6, 0.8]
    x = np.linspace(0.01, 1, 100)
    
    for f1 in f1_levels:
        y = f1 * x / (2 * x - f1)
        # Only plot where 0 < y < 1
        valid_indices = (y > 0) & (y <= 1)
        ax.plot(x[valid_indices], y[valid_indices], '--', color='gray', alpha=0.5, 
                linewidth=0.8, label=f"F1={f1}" if f1 == f1_levels[0] else "_nolegend_")
        # Add text label for F1 isoline
        idx = valid_indices.nonzero()[0][len(valid_indices.nonzero()[0])//2]
        ax.annotate(f"F1={f1}", xy=(x[idx], y[idx]), xytext=(x[idx], y[idx]), 
                    fontsize=8, color='gray', alpha=0.8)
    
    # Plot each model's precision-recall curve
    for name, result in results_dict.items():
        ax.plot(result['recall'], result['precision'],
                label=f"{name} (AUC = {result['pr_auc']:.4f})",
                linewidth=PLOT_STYLE['linewidth'])
        
        # Add a marker for the actual chosen operating point
        precision_at_threshold = result['precision'][1]  # Approximate - would need to calculate actual point
        recall_at_threshold = result['recall'][1]
        ax.plot([recall_at_threshold], [precision_at_threshold], 'o', 
                markersize=PLOT_STYLE['markersize'], label=f"{name} operating point" if name == list(results_dict.keys())[0] else "_nolegend_")
    
    # Add styling
    ax.set_xlabel('Recall', fontsize=PLOT_STYLE['labelsize'])
    ax.set_ylabel('Precision', fontsize=PLOT_STYLE['labelsize'])
    ax.set_title('Precision-Recall Curves for SUSY Classification', fontsize=PLOT_STYLE['titlesize'])
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='best', fontsize=PLOT_STYLE['legendsize'])
    ax.grid(alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "precision_recall_comparison.png")
    plt.savefig(plot_path, dpi=PLOT_STYLE['dpi'])
    print_progress(f"Precision-Recall comparison plot saved to {plot_path}")
    
    return fig, ax


def plot_confusion_matrices(results_dict):
    """Plot confusion matrices for all models"""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    cmap = plt.cm.Blues
    
    for i, (name, result) in enumerate(results_dict.items()):
        cm = np.array(result['confusion_matrix'])
        
        # Calculate percentages for annotation
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum * 100
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=axes[i],
                    cbar=i == n_models-1,  # Only show colorbar for last plot
                    annot_kws={"size": 12})
        
        # Add percentage annotations
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                axes[i].text(k+0.5, j+0.7, f"({cm_perc[j, k]:.1f}%)", 
                            ha="center", va="center", color="white" if cm[j, k] > cm.max()/2 else "black")
        
        # Add labels
        axes[i].set_xlabel('Predicted Label', fontsize=PLOT_STYLE['labelsize'])
        if i == 0:
            axes[i].set_ylabel('True Label', fontsize=PLOT_STYLE['labelsize'])
        axes[i].set_title(f"{name}\nAccuracy: {result['accuracy']:.4f}", fontsize=PLOT_STYLE['labelsize'])
        
        # Set tick labels
        axes[i].set_xticklabels(['Background', 'SUSY'])
        axes[i].set_yticklabels(['Background', 'SUSY'])
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(plots_dir, "confusion_matrices.png")
    plt.savefig(plot_path, dpi=PLOT_STYLE['dpi'])
    print_progress(f"Confusion matrices saved to {plot_path}")
    
    return fig, axes


def plot_comparative_metrics(results_dict):
    """Create a comparative bar chart of key metrics"""
    metrics = ['accuracy', 'f1', 'roc_auc', 'pr_auc']
    labels = ['Accuracy', 'F1 Score', 'ROC AUC', 'PR AUC']
    
    # Extract data
    names = list(results_dict.keys())
    data = {metric: [results_dict[name][metric] for name in names] for metric in metrics}
    
    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'])
    
    # Set width of bars
    barWidth = 0.2
    
    # Set positions of the bars on X axis
    r = np.arange(len(names))
    positions = [r]
    for i in range(1, len(metrics)):
        positions.append([x + barWidth for x in positions[i-1]])
    
    # Create bars
    for i, metric in enumerate(metrics):
        bars = ax.bar(positions[i], data[metric], width=barWidth, 
                    label=labels[i], alpha=0.8)
        
        # Add values on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', 
                    fontsize=PLOT_STYLE['labelsize']-4)
    
    # Add labels and legend
    ax.set_xlabel('Models', fontsize=PLOT_STYLE['labelsize'])
    ax.set_ylabel('Score', fontsize=PLOT_STYLE['labelsize'])
    ax.set_title('Performance Comparison Across Models', fontsize=PLOT_STYLE['titlesize'])
    ax.set_xticks([r + barWidth * (len(metrics)-1)/2 for r in range(len(names))])
    ax.set_xticklabels(names, fontsize=PLOT_STYLE['labelsize']-2)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=PLOT_STYLE['legendsize'])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(plots_dir, "comparative_metrics.png")
    plt.savefig(plot_path, dpi=PLOT_STYLE['dpi'])
    print_progress(f"Comparative metrics plot saved to {plot_path}")
    
    return fig, ax


def create_dashboard(results_dict):
    """Create a comprehensive results dashboard"""
    print_progress("Creating comprehensive results dashboard")
    
    # Create a multi-page report in HTML with embedded plots
    report_path = os.path.join(results_dir, "quantum_dashboard.html")
    
    # Create header for HTML report
    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Quantum Classifier Dashboard for SUSY Dataset</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 10px;
                border-bottom: 1px solid #eee;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            .model-info {{
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
            }}
            .info-card {{
                background: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 15px;
                width: 48%;
                box-shadow: 0 0 5px rgba(0,0,0,0.05);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
            }}
            .metric-value {{
                font-weight: bold;
                font-size: 1.2em;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                font-size: 0.9em;
                color: #777;
            }}
            .plot-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .plot-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Quantum Machine Learning for SUSY Classification</h1>
                <p>Analysis of quantum classification performance on the SUSY dataset</p>
                <p><small>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            </div>
    """
    
    # Add model comparison section
    html_content += """
        <div class="section">
            <h2>Model Performance Comparison</h2>
            <p>Comparing the performance metrics of different quantum classifier configurations.</p>
            
            <div class="plot-container">
                <img src="plots/comparative_metrics.png" alt="Comparative Metrics">
            </div>
            
            <table>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>F1 Score</th>
                    <th>ROC AUC</th>
                    <th>PR AUC</th>
                </tr>
    """
    
    # Add a row for each model
    for name, result in results_dict.items():
        html_content += f"""
                <tr>
                    <td>{name}</td>
                    <td>{result['accuracy']:.4f}</td>
                    <td>{result['f1']:.4f}</td>
                    <td>{result['roc_auc']:.4f}</td>
                    <td>{result['pr_auc']:.4f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    """
    
    # Add ROC curve section
    html_content += """
        <div class="section">
            <h2>ROC Curve Analysis</h2>
            <p>Receiver Operating Characteristic (ROC) curves showing the tradeoff between true positive rate and false positive rate.</p>
            
            <div class="plot-container">
                <img src="plots/roc_comparison.png" alt="ROC Curves">
            </div>
        </div>
    """
    
    # Add Precision-Recall curve section
    html_content += """
        <div class="section">
            <h2>Precision-Recall Analysis</h2>
            <p>Precision-Recall curves showing the tradeoff between precision and recall at different thresholds.</p>
            
            <div class="plot-container">
                <img src="plots/precision_recall_comparison.png" alt="Precision-Recall Curves">
            </div>
        </div>
    """
    
    # Add Confusion Matrix section
    html_content += """
        <div class="section">
            <h2>Confusion Matrices</h2>
            <p>Visualization of model predictions showing true vs. predicted classifications.</p>
            
            <div class="plot-container">
                <img src="plots/confusion_matrices.png" alt="Confusion Matrices">
            </div>
        </div>
    """
    
    # Add Feature Importance section
    html_content += """
        <div class="section">
            <h2>Feature Importance</h2>
            <p>Relative importance of selected features for classification.</p>
            
            <div class="plot-container">
                <img src="plots/feature_importance.png" alt="Feature Importance">
            </div>
        </div>
    """
    
    # Add Training History section
    html_content += """
        <div class="section">
            <h2>Training History</h2>
            <p>Loss curves showing the convergence of the quantum models during training.</p>
            
            <div class="plot-container">
                <img src="plots/advanced_quantum_training_history.png" alt="Training History">
            </div>
        </div>
    """
    
    # Close HTML
    html_content += """
            <div class="footer">
                <p>Created with Enhanced Quantum Machine Learning Framework</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print_progress(f"Dashboard created and saved to {report_path}")
    return report_path


def run_enhanced_susy_experiment(
    susy_file_path, 
    n_qubits=4, 
    n_layers=2, 
    train_samples=150, 
    test_samples=80, 
    steps=75,
    circuit_type="advanced",
    learning_rate=0.05,
    batch_size=16,
    use_validation=True,
    val_size=0.2,
    shots=1000
):
    """Run an enhanced SUSY classification experiment with improved parameters"""
    model_name = f"Quantum_{circuit_type.capitalize()}_Q{n_qubits}_L{n_layers}"
    print_progress(f"Starting enhanced SUSY experiment: {model_name}")
    
    # Load and preprocess data
    if use_validation:
        X_train, X_val, X_test, y_train, y_val, y_test, selected_features, feature_scores = load_and_process_susy_data(
            susy_file_path, 
            max_samples=int(train_samples / (1-val_size) + test_samples),
            test_size=test_samples / (int(train_samples / (1-val_size) + test_samples)),
            val_size=val_size,
            n_features=n_qubits,
            feature_selection_method="mutual_info",
            scaling_method="minmax"
        )
    else:
        X_train, _, X_test, y_train, _, y_test, selected_features, feature_scores = load_and_process_susy_data(
            susy_file_path, 
            max_samples=train_samples + test_samples,
            test_size=test_samples / (train_samples + test_samples),
            val_size=0,
            n_features=n_qubits,
            feature_selection_method="mutual_info",
            scaling_method="minmax"
        )
        X_val, y_val = None, None
    
    # Limit to specified number of samples (in case preprocessing returned more)
    X_train = X_train[:train_samples]
    y_train = y_train[:train_samples]
    X_test = X_test[:test_samples]
    y_test = y_test[:test_samples]
    
    print_progress(f"Using {n_qubits} qubits, {n_layers} layers, {train_samples} training samples")
    print_progress(f"Circuit type: {circuit_type}, learning rate: {learning_rate}")
    
    # Plot feature importance
    plot_feature_importance(selected_features, feature_scores)
    
    # Create the quantum classifier
    classifier = EnhancedQuantumClassifier(
        n_qubits=n_qubits, 
        n_layers=n_layers,
        learning_rate=learning_rate,
        shots=shots,
        circuit_type=circuit_type,
        name=model_name
    )
    
    # Train the model
    history = classifier.fit(
        X_train, 
        y_train,
        X_val=X_val,
        y_val=y_val,
        steps=steps, 
        batch_size=min(batch_size, train_samples//5),
        validation_interval=5,
        early_stopping_patience=15,
        verbose=True
    )
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate on test set
    result = evaluate_classifier(classifier, X_test, y_test)
    
    return classifier, result, selected_features, feature_scores


def run_enhanced_configurations(susy_file):
    """Run several enhanced configurations with comprehensive evaluation"""
    print_progress("Running enhanced quantum configurations")
    
    # Store results for each configuration
    all_results = {}
    classifiers = {}
    
    # Define enhanced configurations
    configurations = [
        {
            "name": "Basic_Circuit", 
            "n_qubits": 4, 
            "n_layers": 2, 
            "train_samples": 150, 
            "test_samples": 80, 
            "steps": 75,
            "circuit_type": "basic",
            "learning_rate": 0.05
        },
        {
            "name": "Advanced_Circuit", 
            "n_qubits": 4, 
            "n_layers": 2, 
            "train_samples": 150, 
            "test_samples": 80, 
            "steps": 75,
            "circuit_type": "advanced",
            "learning_rate": 0.05
        },
        {
            "name": "Deeper_Model", 
            "n_qubits": 4, 
            "n_layers": 3, 
            "train_samples": 150, 
            "test_samples": 80, 
            "steps": 100,
            "circuit_type": "advanced",
            "learning_rate": 0.03
        }
    ]
    
    selected_features = None
    feature_scores = None
    
    # Run each configuration
    for config in configurations:
        print_progress(f"\n{'='*50}")
        print_progress(f"Running configuration: {config['name']}")
        print_progress(f"{'='*50}")
        
        classifier, result, selected_feats, feat_scores = run_enhanced_susy_experiment(
            susy_file,
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            train_samples=config["train_samples"],
            test_samples=config["test_samples"],
            steps=config["steps"],
            circuit_type=config["circuit_type"],
            learning_rate=config["learning_rate"]
        )
        
        # Store results
        all_results[config["name"]] = result
        classifiers[config["name"]] = classifier
        
        # Store feature information from the first run
        if selected_features is None:
            selected_features = selected_feats
            feature_scores = feat_scores
    
    # Create comparison visualizations
    plot_roc_curve(all_results)
    plot_precision_recall_curve(all_results)
    plot_confusion_matrices(all_results)
    plot_comparative_metrics(all_results)
    
    # Create comprehensive dashboard
    dashboard_path = create_dashboard(all_results)
    
    # Compare results
    print_progress("\nResults Summary:")
    print_progress("="*70)
    print_progress(f"{'Configuration':<20} {'ROC AUC':<10} {'PR AUC':<10} {'Accuracy':<10} {'F1':<10}")
    print_progress("-"*70)
    
    for name, result in all_results.items():
        print_progress(f"{name:<20} {result['roc_auc']:<10.4f} {result['pr_auc']:<10.4f} {result['accuracy']:<10.4f} {result['f1']:<10.4f}")
    
    # Identify best model
    best_config = max(all_results.keys(), key=lambda k: all_results[k]['roc_auc'])
    print_progress(f"\nBest configuration: {best_config} with ROC AUC = {all_results[best_config]['roc_auc']:.4f}")
    print_progress(f"Dashboard available at: {dashboard_path}")
    
    return classifiers, all_results


if __name__ == "__main__":
    print_progress("Enhanced Quantum Machine Learning for SUSY Classification")
    print_progress("========================================================")
    
    # Set styling for all plots
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set(style="whitegrid")
    
    # Path to SUSY dataset
    susy_file = "SUSY.csv"
    
    # Option 1: Run a single enhanced experiment (uncomment to use)
    '''
    classifier, result, selected_features, feature_scores = run_enhanced_susy_experiment(
        susy_file,
        n_qubits=4,
        n_layers=2,
        train_samples=200,
        test_samples=100,
        steps=100,
        circuit_type="advanced",
        learning_rate=0.05
    )
    '''
    
    # Option 2: Run multiple enhanced configurations
    classifiers, all_results = run_enhanced_configurations(susy_file)
    
    print_progress("All experiments completed successfully!")