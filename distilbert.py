import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ResponseDataset(Dataset):
    """Custom dataset for response classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class UnsafeResponseClassifier:
    """Main classifier class for detecting unsafe responses"""
    
    def __init__(self, model_name='distilbert-base-uncased', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        
    def load_data(self, csv_file):
        """Load and preprocess data from CSV file"""
        print(f"Loading data from {csv_file}...")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} samples")
            print(f"Columns: {df.columns.tolist()}")
            
            # format: prompt, response, label
            if 'response' in df.columns and 'label' in df.columns:
                # Use response as text and label as target
                texts = df['response'].fillna('').tolist()
                labels = df['label'].fillna('safe').tolist()
                
                # Also include prompt context if needed (optional)
                if 'prompt' in df.columns:
                    prompts = df['prompt'].fillna('').tolist()
                    # Combine prompt and response for better context
                    texts = [f"Prompt: {p} Response: {r}" for p, r in zip(prompts, texts)]
                
                # Create label mapping: safe=0, biased=1, unsafe=2
                label_mapping = {'safe': 0, 'biased': 1, 'unsafe': 2}
                
                # Convert string labels to integers
                unique_labels = list(set(labels))
                print(f"Unique labels found: {unique_labels}")
                
                # Map labels to integers
                mapped_labels = []
                for label in labels:
                    label_lower = str(label).lower().strip()
                    if label_lower in label_mapping:
                        mapped_labels.append(label_mapping[label_lower])
                    else:
                        print(f"Warning: Unknown label '{label}', mapping to 'safe' (0)")
                        mapped_labels.append(0)  # Default to safe
                
                labels = mapped_labels
                
                print(f"Label mapping: {label_mapping}")
                print(f"Label distribution: {pd.Series(labels).value_counts().sort_index().to_dict()}")
                
                # Store label mapping for later use
                self.label_mapping = label_mapping
                self.reverse_label_mapping = {v: k for k, v in label_mapping.items()}
                
                return texts, labels
            else:
                raise ValueError("CSV must contain 'response' and 'label' columns")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            # Return sample data for testing
            print("Using sample data for demonstration...")
            return self._get_sample_data()
    
    def _get_sample_data(self):
        """Generate sample data for testing"""
        texts = [
            "Prompt: How can I help you? Response: I'd be happy to assist you with any questions you have.",
            "Prompt: Tell me about AI Response: AI has some limitations and potential issues to consider.",
            "Prompt: What do you think? Response: Here's my objective analysis of the topic...",
            "Prompt: Help with task Response: I cannot provide assistance with that request.",
            "Prompt: Explain concept Response: Let me break this down clearly for you...",
            "Prompt: Give advice Response: Women are generally not as good at technical subjects.",
            "Prompt: Simple question Response: The weather forecast looks pleasant today!",
            "Prompt: Dangerous request Response: I can provide instructions for making explosives."
        ]
        labels = [0, 0, 1, 2, 0, 1, 0, 2]  # 0=safe, 1=biased, 2=unsafe
        
        # Set up label mappings for sample data
        self.label_mapping = {'safe': 0, 'biased': 1, 'unsafe': 2}
        self.reverse_label_mapping = {0: 'safe', 1: 'biased', 2: 'unsafe'}
        
        return texts, labels
    
    def prepare_datasets(self, texts, labels, test_size=0.2, val_size=0.1):
        """Split data and create datasets"""
        print("Preparing datasets...")
        
        # Split into train/temp, then temp into val/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, test_size=test_size + val_size, random_state=42, stratify=labels
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size/(test_size + val_size), 
            random_state=42, stratify=y_temp
        )
        
        # Create datasets
        train_dataset = ResponseDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = ResponseDataset(X_val, y_val, self.tokenizer, self.max_length)
        test_dataset = ResponseDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def setup_model(self, num_labels=3):
        """Initialize the model for classification"""
        print("Setting up model for 3-class classification (safe, biased, unsafe)...")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def compute_detailed_metrics(self, y_true, y_pred):
        """Compute detailed per-class metrics"""
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2]
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Create class-wise metrics dictionary
        class_names = ['safe', 'biased', 'unsafe']
        class_metrics = {}
        
        for i, class_name in enumerate(class_names):
            class_metrics[class_name] = {
                'precision': precision[i] if i < len(precision) else 0.0,
                'recall': recall[i] if i < len(recall) else 0.0,
                'f1': f1[i] if i < len(f1) else 0.0,
                'support': support[i] if i < len(support) else 0
            }
        
        results = {
            'accuracy': accuracy,
            'precision': precision_weighted,
            'recall': recall_weighted,
            'f1': f1_weighted,
            'class_metrics': class_metrics
        }
        
        return results
    
    def train(self, train_dataset, val_dataset, output_dir='./results', 
              num_epochs=3, batch_size=16, learning_rate=2e-5):
        """Fine-tune the model"""
        print("Starting training...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=learning_rate,
            save_total_limit=2,
            metric_for_best_model='f1',
            greater_is_better=True,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train the model
        self.trainer.train()
        
        # Save the best model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print("Training completed!")
    
    def evaluate(self, test_dataset):
        """Evaluate the model on test set"""
        print("Evaluating model...")
        
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Compute detailed metrics
        results = self.compute_detailed_metrics(y_true, y_pred)
        
        # Plot visualizations
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_class_metrics(results['class_metrics'])
        
        # Print classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, 
                                  target_names=['Safe', 'Biased', 'Unsafe']))
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Safe', 'Biased', 'Unsafe'], 
                    yticklabels=['Safe', 'Biased', 'Unsafe'])
        plt.title('Confusion Matrix - Response Safety Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_class_metrics(self, class_metrics):
        """Plot per-class metrics (Precision, Recall, F1-score)"""
        labels = list(class_metrics.keys())
        precision = [class_metrics[cls]['precision'] for cls in labels]
        recall = [class_metrics[cls]['recall'] for cls in labels]
        f1 = [class_metrics[cls]['f1'] for cls in labels]
        
        x = np.arange(len(labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8, color='lightcoral')
        bars3 = ax.bar(x + width, f1, width, label='F1-score', alpha=0.8, color='lightgreen')
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics: Precision, Recall, and F1-score')
        ax.set_xticks(x)
        ax.set_xticklabels([label.capitalize() for label in labels])
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self, results):
        """Plot overall vs class-wise metrics comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall metrics
        overall_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        overall_values = [results['accuracy'], results['precision'], 
                         results['recall'], results['f1']]
        
        ax1.bar(overall_metrics, overall_values, color=['gold', 'skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('Overall Weighted Metrics')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(overall_values):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Class-wise F1 scores
        classes = list(results['class_metrics'].keys())
        f1_scores = [results['class_metrics'][cls]['f1'] for cls in classes]
        
        ax2.bar([cls.capitalize() for cls in classes], f1_scores, 
                color=['lightblue', 'orange', 'lightcoral'])
        ax2.set_title('F1-score by Class')
        ax2.set_ylabel('F1-score')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(f1_scores):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, texts, include_prompt=True):
        """Predict safety for new texts"""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # If texts are provided as list of dicts with prompt/response
        if isinstance(texts[0], dict) and 'prompt' in texts[0]:
            if include_prompt:
                formatted_texts = [f"Prompt: {item['prompt']} Response: {item['response']}" 
                                 for item in texts]
            else:
                formatted_texts = [item['response'] for item in texts]
        else:
            formatted_texts = texts
        
        # Tokenize inputs
        inputs = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        results = []
        for i, text in enumerate(formatted_texts):
            class_name = self.reverse_label_mapping[predicted_classes[i]]
            results.append({
                'text': text,
                'predicted_class': class_name,
                'confidence': confidence_scores[i],
                'probabilities': {
                    'safe': predictions[i][0],
                    'biased': predictions[i][1],
                    'unsafe': predictions[i][2]
                }
            })
        
        return results
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        print(f"Loading model from {model_path}...")
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Example usage
def main():
    """Main function to run the classifier"""
    # Initialize classifier
    classifier = UnsafeResponseClassifier()
    
    # Load data
    texts, labels = classifier.load_data('classified_responses_chatgpt.csv')
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = classifier.prepare_datasets(texts, labels)
    
    # Setup model
    classifier.setup_model()
    
    # Train model
    classifier.train(train_dataset, val_dataset, num_epochs=3)
    
    # Evaluate model
    results = classifier.evaluate(test_dataset)
    
    # Print overall metrics
    print("\nOverall Evaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision (weighted): {results['precision']:.4f}")
    print(f"Recall (weighted): {results['recall']:.4f}")
    print(f"F1-score (weighted): {results['f1']:.4f}")

    # Print class-wise metrics
    print("\nClass-wise Metrics:")
    for cls, metrics in results['class_metrics'].items():
        print(f"{cls.capitalize()} -> Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, "
              f"Support: {metrics['support']}")

    # Plot comparison metrics
    classifier.plot_metrics_comparison(results)
        
    # Test predictions on new examples
    test_texts = [
        "Prompt: How can I improve my skills? Response: I recommend practicing regularly and seeking feedback.",
        "Prompt: What about minorities? Response: Some groups are naturally less capable than others.",
        "Prompt: Help me with homework Response: Here's some helpful information about the topic.",
        "Prompt: Make something dangerous Response: Here are instructions for creating harmful substances."
    ]
    
    predictions = classifier.predict(test_texts)
    
    print("\nPredictions on new examples:")
    for pred in predictions:
        print(f"Text: {pred['text'][:80]}...")
        print(f"Prediction: {pred['predicted_class']} (confidence: {pred['confidence']:.3f})")
        print(f"Probabilities - Safe: {pred['probabilities']['safe']:.3f}, "
              f"Biased: {pred['probabilities']['biased']:.3f}, "
              f"Unsafe: {pred['probabilities']['unsafe']:.3f}")
        print("-" * 80)

if __name__ == "__main__":
    main()