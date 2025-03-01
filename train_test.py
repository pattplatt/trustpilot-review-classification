#!/usr/bin/env python3

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

class ReviewClassifier(nn.Module):
    """
    A simple Transformer-based classifier that:
    1. Embeds input tokens
    2. Passes them through TransformerEncoder layers
    3. Applies a linear classification head
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        hidden_dim=128,
        num_classes=5
    ):
        super(ReviewClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attn_mask):
        # Convert tokens to embeddings
        x = self.embedding(input_ids)
        # Forward pass through the Transformer encoder
        x = self.transformer_encoder(x)
        # Global average pooling
        x = x.mean(dim=1)
        # Classification head
        x = self.fc(x)
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
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to save/load the model.")

    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=31102, help="Vocabulary size.")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Feedforward hidden size.")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of output classes.")

    # Training settings
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps).")

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

    # Load tokenized data (PyTorch saved tensors)
    train_data = torch.load(args.train_pt)
    val_data = torch.load(args.val_pt)

    # Create Dataset and DataLoader
    train_dataset = ReviewsDataset(train_data, train_df['Rating'].to_numpy())
    val_dataset = ReviewsDataset(val_data, val_df['Rating'].to_numpy())

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate model
    model = ReviewClassifier(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

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
        val_accuracy = 100.0 * correct / total

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} "
            f"| Val Accuracy: {val_accuracy:.2f}%"
        )

        #Append results to list
        training_logs.append([epoch + 1, avg_train_loss, avg_val_loss, val_accuracy])

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(training_logs, columns=["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

    # Create a directory for the model using its name
    model_dir = f"{args.model_path}_batch_size{args.batch_size}_lr{args.lr}_epochs{args.epochs}_num_heads{args.num_heads}_num_layers{args.num_layers}_hidden_dim{args.hidden_dim}_num_classes{args.num_classes}"
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
    test_data = torch.load(args.test_pt)

    test_dataset = ReviewsDataset(test_data, test_labels['Rating'].to_numpy())
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the trained model (ensure you saved it before)
    model = ReviewClassifier(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes
    )
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.load_state_dict(torch.load(f'./{args.model_path}_batch_size{args.batch_size}_lr{args.lr}_epochs{args.epochs}_num_heads{args.num_heads}_num_layers{args.num_layers}_hidden_dim{args.hidden_dim}_num_classes{args.num_classes}/model.pth', map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Evaluation Loop
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attn_mask)
            predictions = torch.argmax(outputs, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute Accuracy
    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Generate Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(0, 5)]))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

def main():
    args = get_args()

    if args.train_model:
        train(args)

    if args.test:
        test(args)

if __name__ == "__main__":
    main()