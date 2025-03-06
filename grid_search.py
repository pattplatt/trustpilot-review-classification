from sklearn.model_selection import ParameterGrid
import os
import subprocess

grid = {
    "embed_dim": [64, 128],  # Keep 2 values instead of 3
    "num_heads": [4, 8],  # Drop 2, as 4 and 8 are most common
    "num_layers": [2, 3],  # Reduce from 3 to 2 values
    "hidden_dim": [128, 256],  # Keep only higher values for larger models
    "lr": [1e-3, 5e-4],  # Reduce to 2 key values
    "batch_size": [32, 64],  # Keep common batch sizes
    "weighted_loss": [True, False],  # Keep binary choice
    "epochs":[3],
    "num_classes":[5]
}

# Track progress
total_configs = len(list(ParameterGrid(grid)))
current_config = 0

print(f"Starting Grid Search with {total_configs} configurations...\n")

for params in ParameterGrid(grid):
    current_config += 1
    model_dir = f"model_embed{params['embed_dim']}_heads{params['num_heads']}_layers{params['num_layers']}_hidden{params['hidden_dim']}_lr{params['lr']}_batch{params['batch_size']}_epochs{params['epochs']}_classes{params['num_classes']}{'_weighted' if params['weighted_loss'] else ''}"

    os.makedirs(model_dir, exist_ok=True)
    
    # Display current parameter set
    print(f"[{current_config}/{total_configs}] Running with: embed_dim={params['embed_dim']}, num_heads={params['num_heads']}, num_layers={params['num_layers']}, hidden_dim={params['hidden_dim']}, lr={params['lr']}, batch_size={params['batch_size']}, weighted_loss={params['weighted_loss']}")
    
    command = [
        "./train_test.py",
        "--train_model",
        "--test",
        "--train_csv", "./data/trustpilot_reviews_inbalanced_5_classes_small/train_inbalanced_5_class_small.csv",
        "--val_csv", "./data/trustpilot_reviews_inbalanced_5_classes_small/val_inbalanced_5_class_small.csv",
        "--train_pt", "./data/trustpilot_reviews_inbalanced_5_classes_small/train_inbalanced_tokenized_5_class_small.pt",
        "--val_pt", "./data/trustpilot_reviews_inbalanced_5_classes_small/val_inbalanced_tokenized_5_class_small.pt",
        "--test_csv", "./data/trustpilot_reviews_inbalanced_5_classes_small/test_inbalanced_5_class_small.csv",
        "--test_pt", "./data/trustpilot_reviews_inbalanced_5_classes_small/test_inbalanced_tokenized_5_class_small.pt",
        "--vocab_size", "31102",
        "--embed_dim", str(params["embed_dim"]),
        "--num_heads", str(params["num_heads"]),
        "--num_layers", str(params["num_layers"]),
        "--hidden_dim", str(params["hidden_dim"]),
        "--num_classes", "5",
        "--batch_size", str(params["batch_size"]),
        "--lr", str(params["lr"]),
        "--epochs", "3",
        "--device", "cuda",
    ]

    if params["weighted_loss"]:
        command.append("--weighted_loss")

    log_file = os.path.join(model_dir, "train_log.txt")

    print(f"Saving log to: {log_file}")

    with open(log_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)

    print(f"Finished configuration [{current_config}/{total_configs}]\n")

print("Grid Search Completed!")