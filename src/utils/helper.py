import os
import yaml
import random
import numpy as np
from pathlib import Path
import torch


# ==============================================================
#                       CONFIG LOADER
# ==============================================================
def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"✅ Loaded configuration from {config_path}")
    return config


# ==============================================================
#                          SEED SETUP
# ==============================================================
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🌱 Random seed set to: {seed}")


# ==============================================================
#                     CHECKPOINT MANAGEMENT
# ==============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    accuracy,
    loss,
    config,
    is_best=False,
    train_losses=None,
    val_losses=None,
    train_accuracies=None,
    val_accuracies=None,
):
    """Save model checkpoint, including training state."""
    train_losses = train_losses or []
    val_losses = val_losses or []
    train_accuracies = train_accuracies or []
    val_accuracies = val_accuracies or []

    # Ensure directory exists
    save_dir = Path(config["Data"]["root"]) / config["About"]["models_folder"]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define filenames
    checkpoint_name = config["About"]["checkpoint_file"].format(epoch, accuracy, loss)
    best_checkpoint_name = config["About"]["best_checkpoint_file"].format(epoch, accuracy, loss)

    checkpoint_path = save_dir / checkpoint_name

    # Construct checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "accuracy": accuracy,
        "loss": loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }

    # Save current checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")

    # Save best model if applicable
    if is_best:
        best_path = save_dir / best_checkpoint_name
        torch.save(checkpoint, best_path)
        print(f"🏆 Best model saved: {best_path}")


def load_checkpoint(config, model, optimizer=None, scheduler=None, scaler=None, test=False, path=None):
    """Load the latest or best model checkpoint."""
    checkpoints_folder = Path(config["Data"]["root"]) / config["About"]["models_folder"]
    if not checkpoints_folder.exists():
        raise FileNotFoundError(f"❌ Checkpoint folder not found: {checkpoints_folder}")

    def get_latest(files_dict):
        """Return latest checkpoint by max epoch."""
        return files_dict[max(files_dict)] if files_dict else None

    # --------------------------------------------------------------
    # Load from a specific path
    # --------------------------------------------------------------
    if path:
        if not Path(path).exists():
            raise FileNotFoundError(f"❌ Specified checkpoint not found: {path}")

        cp = torch.load(path, map_location="cpu")
        model.load_state_dict(cp["model_state_dict"])

        if optimizer and "optimizer_state_dict" in cp:
            optimizer.load_state_dict(cp["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in cp:
            scheduler.load_state_dict(cp["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in cp:
            scaler.load_state_dict(cp["scaler_state_dict"])

        print(f"✅ Loaded checkpoint from: {path}")
        if config["About"].get("preload") == "cont":
            return (
                cp["epoch"],
                cp["accuracy"],
                cp["train_losses"],
                cp["val_losses"],
                cp["train_accuracies"],
                cp["val_accuracies"],
            )
        return None

    # --------------------------------------------------------------
    # Automatically detect latest or best checkpoint
    # --------------------------------------------------------------
    epochs_checkpoints = {}
    epochs_bests = {}

    for f in checkpoints_folder.glob("*.pth"):
        try:
            name = f.name
            epoch = int(name.split("__")[0].split("_")[1])
            if name.startswith("checkpoint_"):
                epochs_checkpoints[epoch] = f
            elif name.startswith("best_"):
                epochs_bests[epoch] = f
        except Exception:
            continue  # Skip malformed filenames

    latest_checkpoint = get_latest(epochs_checkpoints)
    latest_best = get_latest(epochs_bests)

    preload_mode = config["About"].get("preload", "").lower()

    if preload_mode == "best" or test:
        if not latest_best:
            raise FileNotFoundError("❌ No 'best' checkpoint found.")
        cp = torch.load(latest_best, map_location="cpu")
        model.load_state_dict(cp["model_state_dict"])
        print(f"🏅 Loaded BEST checkpoint: {latest_best}")
        return cp["epoch"]

    elif preload_mode == "cont":
        if not latest_checkpoint:
            raise FileNotFoundError("❌ No checkpoint found for continuation.")
        cp = torch.load(latest_checkpoint, map_location="cpu")
        model.load_state_dict(cp["model_state_dict"])

        if optimizer:
            optimizer.load_state_dict(cp["optimizer_state_dict"])
        if scheduler:
            scheduler.load_state_dict(cp["scheduler_state_dict"])
        if scaler:
            scaler.load_state_dict(cp["scaler_state_dict"])

        print(f"🔁 Resumed from checkpoint: {latest_checkpoint}")
        return (
            cp["epoch"],
            cp["accuracy"],
            cp["train_losses"],
            cp["val_losses"],
            cp["train_accuracies"],
            cp["val_accuracies"],
        )

    else:
        print("ℹ️ No checkpoint loading mode selected.")
        return None
