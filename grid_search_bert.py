
from sklearn.model_selection import ParameterGrid
import os
import subprocess

# Only grid search over the specified parameters.
# Note: weighted_loss is now a string with options "none" or "focal".
#grid = {
#    "hidden_dim": [64 ,128, 256],
#    "lr": [2e-3,2e-4, 5e-4],
#    "batch_size": [32, 64, 128],
#    "weighted_loss": ["none", "focal"]
#}

grid = {
    "hidden_dim": [64 ,128, 256],
    "lr": [2e-4, 1e-4],
    "batch_size": [64, 128],
    "weighted_loss": ["none", "focal"]
}

start_from = 24  # Change if you want to skip some configurations.
total_configs = len(list(ParameterGrid(grid)))
current_config = 0

print(f"Starting Grid Search with {total_configs} configurations...\n")

for params in ParameterGrid(grid):
    current_config += 1

    # Create a directory name that includes the grid parameters.
    model_dir = (
            f"inbalanced_model_hidden{params['hidden_dim']}_lr{params['lr']}_batch{params['batch_size']}_"
        f"weighted_loss_{params['weighted_loss']}"
    )

    if current_config < start_from:
        print(f"Skipping configuration [{current_config}/{total_configs}]...")
        continue

    os.makedirs(model_dir, exist_ok=True)

    # Display current parameter set.
    print(
        f"[{current_config}/{total_configs}] Running with: hidden_dim={params['hidden_dim']}, "
        f"lr={params['lr']}, batch_size={params['batch_size']}, weighted_loss={params['weighted_loss']}"
    )

    command = [
        "python3", "./train_test.py",
        "--train_model",
        "--model_type", "bert",
        "--test",
        "--train_csv", "./data_final/inbalanced_5_classes_augmented_23_downsampled/train_inbalanced_5_classes_augmented_23_downsampled.csv",
        "--val_csv",   "./data_final/inbalanced_5_classes_augmented_23_downsampled/val_inbalanced_5_classes_augmented_23_downsampled.csv",
        "--train_pt",  "./data_final/inbalanced_5_classes_augmented_23_downsampled/train_inbalanced_5_classes_augmented_23_downsampled.pt",
        "--val_pt",    "./data_final/inbalanced_5_classes_augmented_23_downsampled/val_inbalanced_5_classes_augmented_23_downsampled.pt",
        "--test_csv",  "./data_final/inbalanced_5_classes_augmented_23_downsampled/test_inbalanced_5_classes_augmented_23_downsampled.csv",
        "--test_pt",   "./data_final/inbalanced_5_classes_augmented_23_downsampled/test_inbalanced_5_classes_augmented_23_downsampled.pt",        
        "--vocab_size", "31102",
        "--embed_dim", "768",    # Fixed value.
        "--num_heads", "8",      # Fixed value.
        "--num_layers", "1",     # Fixed value.
        "--hidden_dim", str(params["hidden_dim"]),
        "--num_classes", "5",    # Fixed value.
        "--batch_size", str(params["batch_size"]),
        "--lr", str(params["lr"]),
        "--epochs", "5",         # Fixed value.
        "--device", "cuda",
    ]

    # Always pass the weighted_loss parameter.
    command.extend(["--weighted_loss", params["weighted_loss"]])

    log_file = os.path.join(model_dir, "train_log.txt")
    print(f"Saving log to: {log_file}")

    with open(log_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)

    print(f"Finished configuration [{current_config}/{total_configs}]\n")

print("Grid Search Completed!")
