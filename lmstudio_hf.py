import json
import os
from pathlib import Path  
import sys
import shutil

def select_models(model_choices):
    # Pre-select already imported models
    selected = [choice[2] for choice in model_choices]  # choice[2] is is_imported
    idx = 0
    window_size = os.get_terminal_size().lines - 5
    
    while True:
        print("\033[H\033[J", end="")
        print("❯ lm-studio - Hugging Face Model Manager \nAvailable models (↑/↓ to navigate, SPACE to select, ENTER to confirm, Ctrl+C to quit):")
        
        window_start = max(0, min(idx - window_size + 3, len(model_choices) - window_size))
        window_end = min(window_start + window_size, len(model_choices))

        for i in range(window_start, window_end):
            display_name, _, _, _ = model_choices[i]
            print(f"{'>' if i == idx else ' '} {'◉' if selected[i] else '○'} {display_name}")

        key = get_key()
        if key == "\x1b[A":  # Up arrow
            idx = max(0, idx - 1)
        elif key == "\x1b[B":  # Down arrow
            idx = min(len(model_choices) - 1, idx + 1)
        elif key == " ":
            selected[idx] = not selected[idx]
        elif key == "\r":  # Enter key
            break
        elif key == "\x03":  # Ctrl+C
            print("\nImport is cancelled. Do nothing.")
            sys.exit(0)

    return [choice for choice, is_selected in zip(model_choices, selected) if is_selected]

def get_key():
    """Get a single keypress from the user."""
    import tty, termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def manage_models():
    "Import models from the Hugging Face cache."
    # Priority: HF_HOME > XDG_CACHE_HOME/huggingface > ~/.cache/huggingface
    if "HF_HOME" in os.environ:
        cache_dir = Path(os.environ["HF_HOME"])
    elif "XDG_CACHE_HOME" in os.environ:
        cache_dir = Path(os.environ["XDG_CACHE_HOME"]) / "huggingface"
    else:
        cache_dir = Path(os.path.expanduser("~/.cache/huggingface"))
    
    # Allow custom LM Studio directory via environment variable
    if "LMSTUDIO_HOME" in os.environ:
        lm_studio_dir = Path(os.environ["LMSTUDIO_HOME"])
    else:
        lm_studio_dir = Path(os.path.expanduser("~/.cache/lm-studio/models"))
    
    found_models = set()
    for root, dirs, _ in os.walk(cache_dir):
        for d in dirs:
            if "--" in d:  # Check for Hugging Face naming pattern (org--model)
                model_dir = Path(root) / d
                snapshots_dir = model_dir / "snapshots"
                if not snapshots_dir.exists():
                    continue

                # Search for config.json in any subfolder under snapshots
                config_found = False
                snapshot_path = None
                for config_root, _, config_files in os.walk(snapshots_dir):
                    if "config.json" in config_files:
                        config_path = Path(config_root) / "config.json"
                        try:
                            with open(config_path) as f:
                                config = json.load(f)
                                model_type = config.get("model_type", "unknown").lower()
                                config_found = True
                                snapshot_path = Path(config_root)
                                break
                        except (json.JSONDecodeError, FileNotFoundError):
                            continue

                if not config_found or not snapshot_path:
                    continue
                    
                parts = d.split("--")
                model_name = "/".join(parts)  # Keep full name including organization
                if model_name:
                    # Store model_type, model_name, and snapshot_path
                    found_models.add((model_type, model_name, snapshot_path))

    if not found_models:
        print("No models found in Hugging Face cache")
        return

    # Create list of models with their current import status
    model_choices = []
    for model_type, model, snapshot_path in sorted(found_models):
        target_path = lm_studio_dir / f"{model}"
        is_imported = target_path.exists()
        status = " (already imported)" if is_imported else ""
        display_name = f"({model_type}) {model}{status}"
        model_choices.append((display_name, model, is_imported, snapshot_path))

    # Show interactive selection menu
    selected = select_models(model_choices)
    print("\nImporting models...\n")
    
    for display_name, model_name, is_imported, snapshot_path in selected:
        target_path = lm_studio_dir / f"{model_name}"
        
        if is_imported:
            # Remove existing directory or symlink
            if target_path.is_symlink() or target_path.exists():
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
            print(f"Removed {model_name}")
        
        else:
            # Create parent directories and target directory
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Create symbolic links for all files in the snapshot directory
            for item in snapshot_path.iterdir():
                link_path = target_path / item.name
                os.symlink(item, link_path)
            
            print(f"Imported {model_name} (symlinked files)")

if __name__ == "__main__":
    manage_models()

