import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    print("Warning: seaborn not available. Using matplotlib only.")
    sns = None
from datetime import datetime

def evaluate_model_on_test(model, test_dataloader, device, class_names=None, save_results=True):
    """
    Evaluate the trained model on test data and generate predictions for confusion matrix.
    
    Args:
        model: Trained PyTorch model
        test_dataloader: DataLoader for test data
        device: Device to run evaluation on
        class_names: List of class names for labeling (optional)
        save_results: Whether to save evaluation results to files
    
    Returns:
        tuple: (true_labels, predictions, test_accuracy, test_loss)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions and labels for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100.0 * correct / total
    average_loss = total_loss / len(test_dataloader)
    
    print(f"Test Results:")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Loss: {average_loss:.4f}")
    print(f"Total Test Samples: {total}")
    print(f"Correct Predictions: {correct}")
    
    # Save evaluation results in multiple formats
    if save_results:
        save_evaluation_results(all_labels, all_predictions, test_accuracy, average_loss, class_names)
    
    return all_labels, all_predictions, test_accuracy, average_loss

def save_evaluation_results(true_labels, predictions, test_accuracy, test_loss, class_names=None):
    """
    Save evaluation results in multiple formats (JSON, pickle, CSV, numpy).
    """
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create class mapping
    unique_labels = sorted(list(set(true_labels + predictions)))
    if class_names is None:
        class_mapping = {i: f'Class {i}' for i in unique_labels}
    else:
        class_mapping = {i: class_names[i] if i < len(class_names) else f'Class {i}' 
                        for i in unique_labels}
    
    # Prepare results dictionary
    results = {
        'true_labels': true_labels,
        'predictions': predictions,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'class_mapping': class_mapping,
        'timestamp': timestamp,
        'total_samples': len(true_labels),
        'correct_predictions': sum(1 for t, p in zip(true_labels, predictions) if t == p)
    }
    
    # 1. Save as JSON
    json_filename = f'evaluation_results_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results to: {json_filename}")
    
    # 2. Save as pickle (preserves exact types)
    pickle_filename = f'evaluation_results_{timestamp}.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ Saved results to: {pickle_filename}")
    
    # 3. Save as CSV
    csv_filename = f'predictions_vs_true_{timestamp}.csv'
    df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': predictions,
        'correct': [t == p for t, p in zip(true_labels, predictions)]
    })
    df.to_csv(csv_filename, index=False)
    print(f"✅ Saved results to: {csv_filename}")
    
    # 4. Save as numpy arrays
    numpy_filename = f'evaluation_arrays_{timestamp}.npz'
    np.savez(numpy_filename, 
             true_labels=np.array(true_labels),
             predictions=np.array(predictions),
             class_mapping=np.array(list(class_mapping.items()), dtype=object))
    print(f"✅ Saved results to: {numpy_filename}")
    
    return timestamp

def load_and_visualize_results(filename, file_type='auto'):
    """
    Load evaluation results from saved files and create visualizations.
    
    Args:
        filename: Path to the saved results file
        file_type: Type of file ('json', 'pickle', 'csv', 'numpy', or 'auto')
    """
    # Auto-detect file type if not specified
    if file_type == 'auto':
        if filename.endswith('.json'):
            file_type = 'json'
        elif filename.endswith('.pkl'):
            file_type = 'pickle'
        elif filename.endswith('.csv'):
            file_type = 'csv'
        elif filename.endswith('.npz'):
            file_type = 'numpy'
        else:
            raise ValueError("Cannot auto-detect file type. Please specify file_type.")
    
    # Load data based on file type
    if file_type == 'json':
        with open(filename, 'r') as f:
            results = json.load(f)
        true_labels = results['true_labels']
        predictions = results['predictions']
        class_mapping = results['class_mapping']
        
    elif file_type == 'pickle':
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        true_labels = results['true_labels']
        predictions = results['predictions']
        class_mapping = results['class_mapping']
        
    elif file_type == 'csv':
        df = pd.read_csv(filename)
        true_labels = df['true_label'].tolist()
        predictions = df['predicted_label'].tolist()
        # Create basic class mapping
        unique_labels = sorted(list(set(true_labels + predictions)))
        class_mapping = {str(i): f'Class {i}' for i in unique_labels}
        
    elif file_type == 'numpy':
        data = np.load(filename, allow_pickle=True)
        true_labels = data['true_labels'].tolist()
        predictions = data['predictions'].tolist()
        # Reconstruct class mapping
        class_mapping_array = data['class_mapping']
        class_mapping = {str(k): v for k, v in class_mapping_array}
    
    # Create visualizations
    visualize_evaluation_results(true_labels, predictions, class_mapping)
    
    return true_labels, predictions, class_mapping

def visualize_evaluation_results(true_labels, predictions, class_mapping):
    """
    Create comprehensive visualizations of evaluation results.
    """
    # Convert class_mapping keys to integers if they're strings
    try:
        class_names = [class_mapping[str(i)] for i in sorted([int(k) for k in class_mapping.keys()])]
    except:
        class_names = list(class_mapping.values())
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Create figure with multiple subplots for comprehensive analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix (Raw Counts)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Predicted', fontsize=12)
    axes[0,0].set_ylabel('True', fontsize=12)
    
    # 2. Normalized Confusion Matrix (Percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0,1])
    axes[0,1].set_title('Normalized Confusion Matrix (%)', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Predicted', fontsize=12)
    axes[0,1].set_ylabel('True', fontsize=12)
    
    # 3. Per-class accuracy visualization
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    axes[1,0].bar(class_names, per_class_accuracy, color='skyblue', alpha=0.7)
    axes[1,0].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Classes', fontsize=12)
    axes[1,0].set_ylabel('Accuracy', fontsize=12)
    axes[1,0].set_ylim(0, 1)
    for i, v in enumerate(per_class_accuracy):
        axes[1,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 4. Class distribution in test set
    unique_labels, counts = np.unique(true_labels, return_counts=True)
    axes[1,1].pie(counts, labels=[class_names[i] for i in unique_labels], autopct='%1.1f%%',
                  startangle=90, colors=plt.cm.Set3.colors)
    axes[1,1].set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print("="*50)
    print(classification_report(true_labels, predictions, target_names=class_names))
    
    # Calculate and display additional metrics
    test_accuracy = 100.0 * sum(1 for t, p in zip(true_labels, predictions) if t == p) / len(true_labels)
    print(f"\nAdditional Metrics:")
    print("="*30)
    print(f"Overall Test Accuracy: {test_accuracy:.2f}%")
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        if i < len(per_class_accuracy):
            print(f"{class_name} Accuracy: {per_class_accuracy[i]:.3f} ({per_class_accuracy[i]*100:.1f}%)")
    
    # Misclassification analysis
    print(f"\nMisclassification Analysis:")
    print("="*35)
    total_samples = len(true_labels)
    correct_predictions = sum(1 for t, p in zip(true_labels, predictions) if t == p)
    misclassified = total_samples - correct_predictions
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Misclassified: {misclassified}")
    print(f"Error Rate: {(misclassified/total_samples)*100:.2f}%")

# Example usage function (replace with your actual model evaluation)
def example_evaluation():
    """
    Example of how to use the evaluation functions.
    Replace this with your actual model, data loader, and device.
    """
    # This is just an example - replace with your actual evaluation
    print("This is an example. Replace with your actual model evaluation code.")
    print("Example usage:")
    print("true_labels, predictions, test_accuracy, test_loss = evaluate_model_on_test(")
    print("    model, test_loader, device, class_names=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']")
    print(")")
    
    # Simulate some example data for demonstration
    np.random.seed(42)
    true_labels = np.random.randint(0, 5, 100).tolist()
    predictions = (np.array(true_labels) + np.random.randint(-1, 2, 100)).clip(0, 4).tolist()
    
    # Save results
    timestamp = save_evaluation_results(
        true_labels, predictions, 85.0, 0.45, 
        class_names=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    )
    
    # Demonstrate loading and visualization
    print(f"\nLoading and visualizing results from saved files...")
    load_and_visualize_results(f'evaluation_results_{timestamp}.json')

if __name__ == "__main__":
    # Run example (replace this with your actual evaluation code)
    example_evaluation()

# Create confusion matrix and visualize with matplotlib
cm = confusion_matrix(true_labels, predictions)

# Define class names (adjust based on your dataset)
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']

# Create figure with multiple subplots for comprehensive analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix (Raw Counts)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
axes[0,0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Predicted', fontsize=12)
axes[0,0].set_ylabel('True', fontsize=12)

# 2. Normalized Confusion Matrix (Percentages)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[0,1])
axes[0,1].set_title('Normalized Confusion Matrix (%)', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Predicted', fontsize=12)
axes[0,1].set_ylabel('True', fontsize=12)

# 3. Per-class accuracy visualization
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
axes[1,0].bar(class_names, per_class_accuracy, color='skyblue', alpha=0.7)
axes[1,0].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Classes', fontsize=12)
axes[1,0].set_ylabel('Accuracy', fontsize=12)
axes[1,0].set_ylim(0, 1)
for i, v in enumerate(per_class_accuracy):
    axes[1,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# 4. Class distribution in test set
unique_labels, counts = np.unique(true_labels, return_counts=True)
axes[1,1].pie(counts, labels=[class_names[i] for i in unique_labels], autopct='%1.1f%%',
              startangle=90, colors=plt.cm.Set3.colors)
axes[1,1].set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Print detailed classification report
print("\nDetailed Classification Report:")
print("="*50)
print(classification_report(true_labels, predictions, target_names=class_names))

# Calculate and display additional metrics
print(f"\nAdditional Metrics:")
print("="*30)
print(f"Overall Test Accuracy: {test_accuracy:.2f}%")
print(f"Overall Test Loss: {test_loss:.4f}")

# Per-class metrics
for i, class_name in enumerate(class_names):
    if i < len(per_class_accuracy):
        print(f"{class_name} Accuracy: {per_class_accuracy[i]:.3f} ({per_class_accuracy[i]*100:.1f}%)")

# Misclassification analysis
print(f"\nMisclassification Analysis:")
print("="*35)
total_samples = len(true_labels)
correct_predictions = sum(1 for t, p in zip(true_labels, predictions) if t == p)
misclassified = total_samples - correct_predictions
print(f"Total Samples: {total_samples}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Misclassified: {misclassified}")
print(f"Error Rate: {(misclassified/total_samples)*100:.2f}%")