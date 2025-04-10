#!/usr/bin/env python3

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModel, AutoConfig

class ReviewsDataset(Dataset):
    """
    A PyTorch Dataset for encoded reviews. 
    Expects the tokenized tensors (input_ids, attention_mask) and
    an array of labels.
    """
    def __init__(self, tokenized_tensors, labels):
        """
        Args:
            tokenized_tensors (dict): A dictionary with 'input_ids' and 'attention_mask' tensors.
            labels (array-like): Numeric labels corresponding to each example.
        """
        self.input_ids = tokenized_tensors["input_ids"]
        self.attn_mask = tokenized_tensors["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            # Example offset for 1–5 labels; adjust as appropriate for your data
            "labels": torch.tensor(self.labels[idx] - 1, dtype=torch.long)
        }

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embedding_dim):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)  # Get sequence length
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        return self.position_embeddings(position_ids)
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss *= self.alpha[targets]
        return focal_loss.mean()

class ReviewClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=768,
        num_heads=4,
        num_layers=2,
        hidden_dim=768,
        num_classes=5,
        max_seq_len=512,
        droput=0.1,
        model_type="vanilla"
    ):
        super(ReviewClassifier, self).__init__()
        self.model_type = model_type
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        if self.model_type == "vanilla":
            # 1) A typical embedding layer
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

            # 2) Positional embedding
            self.positional_embedding = PositionalEmbedding(max_seq_len, embed_dim)

            # 3) Transformer encoder
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
                dropout=droput
            )
            self.transformer_encoder = nn.TransformerEncoder(
                self.encoder_layer,
                num_layers=num_layers
            )
            # Then feed-forward
            self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(droput)
            self.fc2 = nn.Linear(hidden_dim, num_classes)

        elif self.model_type == "bert":
            # Load BERT
            self.bert = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")
            for name, param in self.bert.named_parameters():
                if "encoder.layer.10" in name or "encoder.layer.11" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
           # Classification layers
            # BERT hidden size is typically 768 for bert-base-uncased
            self.fc1 = nn.Linear(768 * 2, hidden_dim)
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(droput)
            self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attn_mask=None):
        if self.model_type == "vanilla":
            # ======= Vanilla approach =======
            x = self.token_embedding(input_ids)
            pos_embed = self.positional_embedding(input_ids)
            x = x + pos_embed  # Add positional embeddings
            x = self.transformer_encoder(x)

            # [CLS] token representation (assuming first token)
            cls_token_output = x[:, 0, :]
            # Max pooling across sequence
            pooled_output, _ = torch.max(x, dim=1)

        elif self.model_type == "bert":
            # ======= BERT approach =======
            outputs = self.bert(input_ids, attention_mask=attn_mask)
            # BERT's last hidden states: (batch_size, seq_len, hidden_size)
            last_hidden_state = outputs.last_hidden_state

            # For typical classification, you can use the [CLS] token
            cls_token_output = last_hidden_state[:, 0, :]
            # Max pooling
            pooled_output, _ = torch.max(last_hidden_state, dim=1)

        # Then combine the two representations
        x = torch.cat((cls_token_output, pooled_output), dim=1)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and test a Transformer classifier on reviews.")

    # Required arguments
    parser.add_argument("--train_model", action="store_true", help="Train the model.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training set CSV.")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation set CSV.")
    parser.add_argument("--train_pt", type=str, required=True, help="Path to tokenized training data (.pt).")
    parser.add_argument("--val_pt", type=str, required=True, help="Path to tokenized validation data (.pt).")

    # Test arguments
    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test set CSV.")
    parser.add_argument("--test_pt", type=str, required=True, help="Path to tokenized test data (.pt).")

    # Model parameters
    parser.add_argument("--model_type",type=str,default="vanilla",choices=["vanilla", "bert", "roberta"],help="Model architecture choice: vanilla, bert, or roberta.")
    parser.add_argument("--vocab_size", type=int, default=31102, help="Vocabulary size.")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Feedforward hidden size.")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of output classes.")
    parser.add_argument("--max_seq_len", default=512, help="Max sequence length of positional encoding")
    parser.add_argument("--dropout", default=0.1, help="Droput value")

    # Training settings
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps).")
    parser.add_argument(
        "--weighted_loss",
        type=str,
        default="none",
        choices=["none", "weighted", "focal"],
        help="Loss function: none (CrossEntropy), weighted (weighted CrossEntropy), or focal (FocalLoss)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a saved model (.pth) file for testing. If None, the code uses the default path based on hyperparameters."
    )

    return parser.parse_args()

def train(args):

    # If device not provided, choose auto
    if args.device is None:
        # If MPS (Metal on Apple Silicon) is available, prefer that, otherwise GPU if available, else CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    training_logs=[]

    print(f"Using device: {device}")

    # Load the CSVs containing the labels
    train_df = pd.read_csv(args.train_csv, delimiter=",", header=0, encoding="utf8")
    val_df = pd.read_csv(args.val_csv, delimiter=",", header=0, encoding="utf8")

    # Get label counts
    label_counts = np.bincount(train_df['Rating'] - 1)  # Count occurrences of each label

    # Normalize to get ratio
    label_ratios = label_counts / sum(label_counts)

    print("Label Ratios:", label_ratios)

    # Compute inverse frequencies
    class_weights = 1.0 / (label_counts + 1e-6)  # Avoid division by zero
    class_weights /= class_weights.sum()  # Normalize

    # Convert to tensor for PyTorch
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Class Weights:", class_weights_tensor)

    # Load tokenized data (PyTorch saved tensors)
    train_data = torch.load(args.train_pt,  weights_only=False)
    val_data = torch.load(args.val_pt,  weights_only=False)

    # Create Dataset and DataLoader
    train_dataset = ReviewsDataset(train_data, train_df['Rating'].to_numpy())
    val_dataset = ReviewsDataset(val_data, val_df['Rating'].to_numpy())

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = ReviewClassifier(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        max_seq_len=args.max_seq_len,
        droput=args.dropout,
        model_type=args.model_type
    ).to(device)

    # Compute inverse frequencies
    class_weights = 1.0 / (label_counts + 1e-6)  # Avoid division by zero
    class_weights /= class_weights.sum()  # Normalize

    # Convert to tensor for PyTorch
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Class Weights:", class_weights_tensor)

    # Loss and optimizer
    if args.weighted_loss == "weighted":
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif args.weighted_loss == "focal":
        criterion = FocalLoss(alpha=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
   
    train_losses = []
    val_losses = []
    
   # Training loop
    for epoch in range(args.epochs):
        # Training Phase
        model.train()
        total_train_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attn_mask)
            #compute weighted loss if args.weighted_loss is set

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        
        avg_train_loss = total_train_loss / len(train_dataloader)        
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attn_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100.0 * correct / total

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} "
            f"| Val Accuracy: {val_accuracy:.2f}%"
        )

        #Append results to list
        training_logs.append([epoch + 1, avg_train_loss, avg_val_loss, val_accuracy])
        scheduler.step(avg_val_loss)

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(training_logs, columns=["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

    # Create a directory for the model using its name
    model_dir = f"model_embed{args.embed_dim}_heads{args.num_heads}_layers{args.num_layers}_hidden{args.hidden_dim}_lr{args.lr}_batch{args.batch_size}_epochs{args.epochs}_classes{args.num_classes}_{args.weighted_loss}"
    os.makedirs(model_dir, exist_ok=True)  # Ensure directory is created

    # Save the trained model inside the directory
    model_save_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Training Complete! Model saved to {model_save_path}")

    # Save training logs as CSV inside the directory
    csv_save_path = os.path.join(model_dir, "training_log.csv")
    df.to_csv(csv_save_path, index=False)
    print(f"Training log saved to {csv_save_path}")

def test(args):

    test_labels = pd.read_csv(args.test_csv, delimiter=",", header=0, encoding="utf8")
    test_data = torch.load(args.test_pt, weights_only=False)

    test_dataset = ReviewsDataset(test_data, test_labels['Rating'].to_numpy())
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = ReviewClassifier(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        max_seq_len=args.max_seq_len,
        droput=args.dropout,
        model_type=args.model_type
    )

    # If device not provided, choose auto
    if args.device is None:
        # If MPS (Metal on Apple Silicon) is available, prefer that, otherwise GPU if available, else CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model_dir = f"model_embed{args.embed_dim}_heads{args.num_heads}_layers{args.num_layers}_hidden{args.hidden_dim}_lr{args.lr}_batch{args.batch_size}_epochs{args.epochs}_classes{args.num_classes}_{args.weighted_loss}"
    
    os.makedirs(model_dir, exist_ok=True)
    if args.model_path is not None:
        model_load_path = args.model_path
    else:
        model_load_path = os.path.join(model_dir, "model.pth")

    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Get label counts
    label_counts = np.bincount(test_labels['Rating'] - 1)  # Count occurrences of each label

    # Normalize to get ratio
    label_ratios = label_counts / sum(label_counts)

    print("Label Ratios:", label_ratios)

    # Compute inverse frequencies
    class_weights = 1.0 / (label_counts + 1e-6)  # Avoid division by zero
    class_weights /= class_weights.sum()  # Normalize

    # Convert to tensor for PyTorch
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Loss and optimizer
    if args.weighted_loss == "weighted":
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif args.weighted_loss == "focal":
        criterion = FocalLoss(alpha=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    # Evaluation Loop
    all_preds = []
    all_labels = []
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attn_mask)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss = test_loss / len(test_dataloader)
    # Compute Accuracy
    test_accuracy = accuracy_score(all_labels, all_preds)
    # Compute F1-score (weighted)
    test_f1_score = f1_score(all_labels, all_preds, average="weighted")

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test F1-Score: {test_f1_score:.4f}")

    # Generate Classification Report
    class_report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(0, 5)], output_dict=True)

    # Convert classification report to DataFrame
    class_report_df = pd.DataFrame(class_report).transpose()
    
    # Save accuracy and classification report to CSV
    test_report_path = os.path.join(model_dir, "test_results.csv")
    
    # Save as CSV
    with open(test_report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy,{test_accuracy * 100:.2f}%\n\n")
        f.write(f"Test F1-Score,{test_f1_score:.4f}\n\n")
        class_report_df.to_csv(f)

    print(f"Test results saved to {test_report_path}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(0, 5), yticklabels=np.arange(0, 5))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    # Save the figure inside the model directory
    conf_matrix_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()

    log_path = os.path.join(model_dir, "training_log.csv")
    if os.path.exists(log_path):
        train_log_df = pd.read_csv(log_path)
        train_losses = train_log_df["Train Loss"].tolist()
        val_losses = train_log_df["Val Loss"].tolist()
    else:
        print("Training_log.csv not found. Skipping loss plot.")

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.axhline(test_loss, color='r', linestyle='--', label=f"Test Loss: {test_loss:.4f}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training, Validation and Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure inside the model directory
    losses_path = os.path.join(model_dir, "losses.png")
    plt.savefig(losses_path)
    plt.close()

def main():
    args = get_args()

    if args.train_model:
        train(args)

    if args.test:
        test(args)

if __name__ == "__main__":
    main()
