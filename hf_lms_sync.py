import argparse
from collections.abc import Iterable
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Tuple, Dict, Any
import urllib.request


@dataclass
class HuggingFaceModel:
    model_type: str
    model_name: str
    snapshot_path: Path
    is_synced_from_lm: bool = False


@dataclass
class LMStudioModel:
    model_name: str
    model_path: Path
    is_synced_from_hf: bool = False


class ModelSyncManager:
    """Manages synchronization of models between Hugging Face cache and LM Studio."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.dry_run_operations: list[str] = []
        self.mode: str = args.mode if hasattr(args, 'mode') else 'symlink'  # Default to symlink

        # Setup logging
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        # Determine directories
        self.hf_cache_dir: Path = self._determine_hf_cache_dir(args)
        self.lm_studio_dir: Path = self._determine_lm_studio_dir(args)

        logging.info(f"Using Hugging Face cache directory: {self.hf_cache_dir}")
        logging.info(f"Using LM Studio models directory: {self.lm_studio_dir}")
        logging.info(f'Using dry run mode: {"true" if self.dry_run else "false"}')
        logging.info(f'Using mode: {self.mode}')

    def _determine_hf_cache_dir(self, args: argparse.Namespace) -> Path:
        """Determine the Hugging Face cache directory."""
        if args.hf_cache_dir:
            return Path(os.path.expanduser(args.hf_cache_dir))
        elif "HF_HOME" in os.environ:
            return Path(os.environ["HF_HOME"])
        elif "XDG_CACHE_HOME" in os.environ:
            return Path(os.environ["XDG_CACHE_HOME"]) / "huggingface"
        else:
            return Path(os.path.expanduser("~/.cache/huggingface"))

    def _determine_lm_studio_dir(self, args: argparse.Namespace) -> Path:
        """Determine the LM Studio models directory."""
        if args.lm_studio_dir:
            return Path(os.path.expanduser(args.lm_studio_dir))
        elif "LMSTUDIO_HOME" in os.environ:
            return Path(os.environ["LMSTUDIO_HOME"])
        else:
            return Path(os.path.expanduser("~/.cache/lmstudio/models"))

    def sync_models(self) -> None:
        """Main synchronization function."""

        # Discover models
        hf_models = self._discover_hf_models()
        existing_lm_models, lm_model_candidates = self._discover_lm_models()

        # Sync Hugging Face models to LM Studio
        self._sync_hf_to_lm(hf_models, existing_lm_models)

        # Sync LM Studio models to Hugging Face
        self._sync_lm_to_hf(lm_model_candidates, hf_models)

        # Show dry run summary if needed
        if self.dry_run:
            self._show_dry_run_summary()

    def _discover_hf_models(self) -> list[HuggingFaceModel]:
        """Discover models in the Hugging Face cache."""
        hf_model_candidates: list[HuggingFaceModel] = []
        discovered_models = set()

        # Look in hub directory if it exists
        hf_hub_dir = self.hf_cache_dir / "hub"
        hf_search_dirs = [hf_hub_dir] if hf_hub_dir.exists() else []
        hf_search_dirs.append(self.hf_cache_dir)

        for search_dir in hf_search_dirs:
            for root, dirs, _ in os.walk(search_dir):
                for d in dirs:
                    # Skip datasets directories
                    if d.startswith("datasets--") or "datasets" in Path(root).parts:
                        continue

                    # Check for models--org--name pattern
                    if d.startswith("models--") and "--" in d:
                        model_dir = Path(root) / d
                        # Extract model name from models--org--name format
                        parts = d.replace("models--", "").split("--")
                        model_name = "/".join(parts)
                    elif "--" in d:  # Check for other HF naming patterns
                        model_dir = Path(root) / d
                        parts = d.split("--")
                        model_name = "/".join(parts)
                    else:
                        continue

                    if model_name in discovered_models:
                        continue

                    # Look for snapshots directory
                    snapshots_dir = model_dir / "snapshots"
                    if not snapshots_dir.exists():
                        continue

                    # Search for snapshots - with or without config.json
                    model_type = "unknown"
                    snapshot_path = None

                    # Try to find the main snapshot using refs/main
                    refs_main_path = model_dir / "refs" / "main"
                    if refs_main_path.exists():
                        with open(refs_main_path) as f:
                            commit_hash = f.read().strip()
                        snapshot_path = snapshots_dir / commit_hash
                        if not snapshot_path.exists():
                            snapshot_path = None

                    # If no main snapshot found, walk through snapshots directory
                    if not snapshot_path:
                        for config_root, snapshot_dirs, config_files in os.walk(snapshots_dir):
                            # Skip the snapshots directory itself, look in subdirectories
                            if config_root == str(snapshots_dir) and snapshot_dirs:
                                continue

                            # If we found files in a snapshot subdirectory, this is a valid model
                            if config_files or snapshot_dirs:
                                snapshot_path = Path(config_root)
                                break

                    if snapshot_path:
                        # Try to find config.json for model type
                        config_path = snapshot_path / "config.json"
                        if config_path.exists():
                            try:
                                with open(config_path) as f:
                                    config = json.load(f)
                                    model_type = config.get("model_type", "unknown").lower()
                            except (json.JSONDecodeError, FileNotFoundError):
                                pass

                    if snapshot_path and model_name:
                        # Check if this model is already a reverse-synced model (symlinks pointing to LM Studio)
                        is_synced_from_lm = False
                        for item in snapshot_path.iterdir():
                            if item.is_symlink():
                                resolved_path = Path(os.path.realpath(item))
                                if self.lm_studio_dir in resolved_path.parents:
                                    logging.debug(
                                        f"Hugging Face model {model_name} is synced from LM Studio: {resolved_path.parent}, skipping")
                                    is_synced_from_lm = True
                                    break

                        if not is_synced_from_lm:
                            # Store model even if no config.json was found
                            hf_model_candidates.append(HuggingFaceModel(
                                model_type=model_type,
                                model_name=model_name,
                                snapshot_path=snapshot_path
                            ))
                            discovered_models.add(model_name)

        return hf_model_candidates

    def _discover_lm_models(self) -> Tuple[list[LMStudioModel], list[LMStudioModel]]:
        """Discover models in the LM Studio directory."""
        existing_lm_models: list[LMStudioModel] = []  # List of LMStudioModel objects
        # Models to sync from LM Studio to Hugging Face
        lm_model_candidates: list[LMStudioModel] = []

        if self.lm_studio_dir.exists():
            # Check for org/model structure (subdirectories)
            for org_dir in self.lm_studio_dir.iterdir():
                if org_dir.is_dir():
                    # Check if this is an org directory with model subdirectories
                    has_subdirs = False
                    try:
                        for model_dir in org_dir.iterdir():
                            if model_dir.is_dir():
                                has_subdirs = True
                                # This is org/model format
                                model_name = f"{org_dir.name}/{model_dir.name}"
                                is_synced_from_hf = self._is_lm_model_synced_from_hf(model_dir)
                                lm_studio_model = LMStudioModel(
                                    model_name=model_name,
                                    model_path=model_dir,
                                    is_synced_from_hf=is_synced_from_hf
                                )
                                existing_lm_models.append(lm_studio_model)
                                if not is_synced_from_hf:
                                    lm_model_candidates.append(lm_studio_model)
                    except:
                        pass  # Handle permission errors

                    # If no subdirectories, this might be a direct model directory
                    if not has_subdirs:
                        model_name = org_dir.name
                        is_synced_from_hf = self._is_lm_model_synced_from_hf(org_dir)
                        lm_studio_model = LMStudioModel(
                            model_name=model_name,
                            model_path=org_dir,
                            is_synced_from_hf=is_synced_from_hf
                        )
                        existing_lm_models.append(lm_studio_model)
                        if not is_synced_from_hf:
                            lm_model_candidates.append(lm_studio_model)

        return existing_lm_models, lm_model_candidates

    def _is_lm_model_synced_from_hf(self, model_dir: Path) -> bool:
        """Check if a LM Studio model directory is synced from Hugging Face."""
        for item in model_dir.iterdir():
            if item.is_symlink():
                resolved_path = Path(os.path.realpath(item))
                if self.hf_cache_dir in resolved_path.parents:
                    logging.debug(
                        f"LM Studio model {model_dir.name} is synced from Hugging Face: {resolved_path.parent}, skipping")
                    return True
        return False

    def _sync_hf_to_lm(self, hf_models: list[HuggingFaceModel], existing_lm_models: list[LMStudioModel]) -> None:
        """Sync models from Hugging Face to LM Studio."""
        if not hf_models:
            logging.info("No models found in Hugging Face cache to sync to LM Studio.")
            return

        # Create list of models with their current import status
        hf_to_lm_model_choices: list[Tuple[str, HuggingFaceModel, bool, Path]] = []
        for hf_model in sorted(hf_models, key=lambda x: x.model_name):
            is_present_in_lm = False
            target_lm_path = None

            # Check for the exact model path as it would be created
            for lm_model in existing_lm_models:
                if hf_model.model_name == lm_model.model_name:
                    is_present_in_lm = True
                    target_lm_path = lm_model.model_path
                    break

            # If not found, use default path for new imports or removals
            if target_lm_path is None:
                target_lm_path = self.lm_studio_dir / hf_model.model_name

            status = " (present in LM Studio)" if is_present_in_lm else ""
            display_name = f"({hf_model.model_type}) {hf_model.model_name}{status}"
            hf_to_lm_model_choices.append(
                (display_name, hf_model, is_present_in_lm, target_lm_path))

        # Show interactive selection menu for importing to LM Studio
        if hf_to_lm_model_choices:
            selected_hf_models = select_models(
                hf_to_lm_model_choices, "Export Hugging Face cache to LM Studio models")
            if selected_hf_models:
                logging.info("Syncing models to LM Studio...")
                for display_name, hf_model, is_present_in_lm, target_lm_path in selected_hf_models:
                    if is_present_in_lm:
                        # Remove existing directory or symlink
                        if target_lm_path.is_symlink() or target_lm_path.exists():
                            if target_lm_path.is_dir():
                                self._add_dry_run_operation(
                                    f"Would remove directory: {target_lm_path}")
                                if not self.dry_run:
                                    shutil.rmtree(target_lm_path)
                                    logging.info(
                                        f"Removed directory {hf_model.model_name} from LM Studio")
                            else:
                                self._add_dry_run_operation(
                                    f"Would remove symlink: {target_lm_path}")
                                if not self.dry_run:
                                    target_lm_path.unlink()
                                    logging.info(
                                        f"Removed symlink {hf_model.model_name} from LM Studio")

                    # Create parent directories and target directory
                    self._add_dry_run_operation(f"Would create directory: {target_lm_path}")
                    if not self.dry_run:
                        try:
                            target_lm_path.mkdir(parents=True, exist_ok=True)
                        except PermissionError as e:
                            logging.error(f"Permission denied creating directory {target_lm_path}: {e}")
                            continue

                    # Create links or move files based on mode
                    moved_files_count = 0
                    for item in hf_model.snapshot_path.iterdir():
                        target_path = target_lm_path / item.name
                        # Only create link/move if the target path does not exist or is already a symlink
                        if not target_path.exists() or target_path.is_symlink():
                            if self.mode == 'move':
                                self._add_dry_run_operation(
                                    f"Would move: {item} -> {target_path}")
                                if not self.dry_run:
                                    shutil.move(str(item), str(target_path))
                                    moved_files_count += 1
                            else:  # symlink mode
                                self._add_dry_run_operation(
                                    f"Would create symlink: {target_path} -> {item}")
                                if not self.dry_run:
                                    os.symlink(item, target_path)
                        else:
                            logging.info(
                                f"  - Skipping {item.name}: file already exists in LM Studio")

                    # In move mode, try to clean up the source directory
                    if self.mode == 'move':
                        self._cleanup_hf_model_directory(hf_model, moved_files_count)

                    logging.info(
                        f"  - Successfully synced {hf_model.model_name} to LM Studio at {target_lm_path}")

    def _sync_lm_to_hf(self, lm_model_candidates: Iterable[LMStudioModel], hf_models: Iterable[HuggingFaceModel]) -> None:
        """Sync models from LM Studio to Hugging Face."""
        # Show interactive selection menu for exporting to Hugging Face
        if not lm_model_candidates:
            logging.info("No models found in LM Studio to sync to Hugging Face cache.")
            return

        lm_to_hf_model_choices: list[Tuple[str, LMStudioModel, bool, Path]] = []
        for lm_model in lm_model_candidates:
            is_present_in_hf = False
            hf_model_path = self.hf_cache_dir / "hub" / \
                f"models--{lm_model.model_name.replace('/', '--')}"
            for hf_model in hf_models:
                if lm_model.model_name == hf_model.model_name:
                    is_present_in_hf = True
                    hf_model_path = hf_model.snapshot_path.parent.parent
                    break

            status = " (present in HF)" if is_present_in_hf else ""

            model_type = "unknown"
            config_path = lm_model.model_path / "config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                        model_type = config.get("model_type", "unknown").lower()
                except (json.JSONDecodeError, FileNotFoundError):
                    pass

            display_name = f"({model_type}) {lm_model.model_name}{status}"
            lm_to_hf_model_choices.append((display_name, lm_model, is_present_in_hf, hf_model_path))

        if lm_to_hf_model_choices:
            selected_lm_models = select_models(
                lm_to_hf_model_choices, "Syncing LM Studio models to Hugging Face cache")
            if selected_lm_models:
                for display_name, lm_model, is_present_in_hf, hf_model_path in selected_lm_models:
                    if is_present_in_hf:
                        # Remove existing directory or symlink
                        if hf_model_path.is_symlink() or hf_model_path.exists():
                            if hf_model_path.is_dir():
                                self._add_dry_run_operation(
                                    f"Would remove directory: {hf_model_path}")
                                if not self.dry_run:
                                    shutil.rmtree(hf_model_path)
                                    logging.info(
                                        f"Removed directory {lm_model.model_name} from Hugging Face cache")
                            else:
                                self._add_dry_run_operation(
                                    f"Would remove symlink: {hf_model_path}")
                                if not self.dry_run:
                                    hf_model_path.unlink()
                                    logging.info(
                                        f"Removed symlink {lm_model.model_name} from Hugging Face cache")

                    logging.info(
                        f"Syncing {lm_model.model_name} from LM Studio to Hugging Face cache...")
                    try:
                        # 1. Fetch model info from Hugging Face Hub
                        api_url = f"https://huggingface.co/api/models/{lm_model.model_name}"
                        response = web_fetch(api_url)

                        if "error" in response:
                            logging.error(
                                f"  - Error fetching model info for {lm_model.model_name}: {response['error']}")
                            continue

                        model_info = json.loads(response["content"])
                        commit_hash = model_info.get("sha")
                        if not commit_hash:
                            logging.error(
                                f"  - Could not find commit hash for {lm_model.model_name}")
                            continue

                        # 2. Create Hugging Face cache directory structure
                        self._add_dry_run_operation(f"Would create directory: {hf_model_path}")
                        if not self.dry_run:
                            try:
                                hf_model_path.mkdir(parents=True, exist_ok=True)
                            except PermissionError as e:
                                logging.error(f"Permission denied creating directory {hf_model_path}: {e}")
                                continue
                        self._add_dry_run_operation(
                            f"Would create directory: {hf_model_path / 'refs'}")
                        if not self.dry_run:
                            try:
                                (hf_model_path / "refs").mkdir(exist_ok=True)
                            except PermissionError as e:
                                logging.error(f"Permission denied creating directory {hf_model_path / 'refs'}: {e}")
                                continue
                        snapshots_dir = hf_model_path / "snapshots"
                        self._add_dry_run_operation(f"Would create directory: {snapshots_dir}")
                        if not self.dry_run:
                            try:
                                snapshots_dir.mkdir(exist_ok=True)
                            except PermissionError as e:
                                logging.error(f"Permission denied creating directory {snapshots_dir}: {e}")
                                continue
                        snapshot_path = snapshots_dir / commit_hash
                        self._add_dry_run_operation(f"Would create directory: {snapshot_path}")
                        if not self.dry_run:
                            try:
                                snapshot_path.mkdir(exist_ok=True)
                            except PermissionError as e:
                                logging.error(f"Permission denied creating directory {snapshot_path}: {e}")
                                continue

                        # 3. Create refs/main file
                        self._add_dry_run_operation(
                            f"Would write file: {hf_model_path / 'refs' / 'main'}")
                        if not self.dry_run:
                            with open(hf_model_path / "refs" / "main", "w") as f:
                                f.write(commit_hash)

                        # 4. Create blobs and snapshot symlinks
                        blobs_dir = hf_model_path / "blobs"
                        self._add_dry_run_operation(f"Would create directory: {blobs_dir}")
                        if not self.dry_run:
                            try:
                                blobs_dir.mkdir(exist_ok=True)
                            except PermissionError as e:
                                logging.error(f"Permission denied creating directory {blobs_dir}: {e}")
                                continue

                        # Get the list of model files from Hugging Face API
                        model_files_from_api = set()
                        for file_info in model_info.get("siblings", []):
                            file_name = file_info.get("rfilename")
                            if file_name:
                                model_files_from_api.add(file_name)

                        # Identify model files vs user-added files
                        model_files = []
                        user_files = []

                        for item in lm_model.model_path.iterdir():
                            if item.is_file():
                                if item.name in model_files_from_api:
                                    model_files.append(item)
                                else:
                                    user_files.append(item)

                        # Log user files that are being preserved
                        for item in user_files:
                            logging.info(f"    - Preserving user file: {item.name}")

                        # Process model files
                        for item in model_files:
                            # Calculate SHA256 hash of the file
                            with open(item, "rb") as f:
                                sha256_hash = hashlib.sha256(f.read()).hexdigest()

                            # Create blob link or move based on mode
                            blob_path = blobs_dir / sha256_hash
                            # Only create link/move if the target path does not exist or is already a symlink
                            if not blob_path.exists() or blob_path.is_symlink():
                                if self.mode == 'move':
                                    self._add_dry_run_operation(
                                        f"Would move: {item} -> {blob_path}")
                                    if not self.dry_run:
                                        shutil.move(str(item), str(blob_path))
                                else:  # symlink mode
                                    self._add_dry_run_operation(
                                        f"Would create symlink: {blob_path} -> {item}")
                                    if not self.dry_run:
                                        os.symlink(item, blob_path)
                            else:
                                logging.info(
                                    f"    - Skipping blob for {item.name}: file already exists in Hugging Face blobs")

                            # Create snapshot symlink (always symlink to blob, not affected by mode)
                            snapshot_link_path = snapshot_path / item.name
                            # Only create symlink if the target path does not exist or is already a symlink
                            if not snapshot_link_path.exists() or snapshot_link_path.is_symlink():
                                self._add_dry_run_operation(
                                    f"Would create symlink: {snapshot_link_path} -> {blob_path}")
                                if not self.dry_run:
                                    os.symlink(blob_path, snapshot_link_path)
                            else:
                                logging.info(
                                    f"    - Skipping snapshot symlink for {item.name}: non-symlink file already exists in Hugging Face snapshots")

                        # In move mode, try to clean up the source directory
                        if self.mode == 'move':
                            self._cleanup_lm_model_directory(lm_model)

                        # 5. Create config.json and other small metadata files from Hub
                        large_file_extensions = [".gguf", ".safetensors"]
                        for file_info in model_info.get("siblings", []):
                            file_name = file_info.get("rfilename")
                            if file_name and not any(file_name.endswith(ext) for ext in large_file_extensions):
                                snapshot_file_path = snapshot_path / file_name
                                # Skip if a symlink with the same name already exists
                                if snapshot_file_path.is_symlink():
                                    continue

                                file_url = f"https://huggingface.co/{lm_model.model_name}/resolve/main/{file_name}"
                                file_response = web_fetch(file_url)
                                if "content" in file_response:
                                    # Calculate SHA256 hash of the file content
                                    sha256_hash = hashlib.sha256(
                                        file_response["content"].encode('utf-8')).hexdigest()
                                    blob_path = blobs_dir / sha256_hash

                                    # Write the file content to the blob
                                    if not blob_path.exists():
                                        self._add_dry_run_operation(
                                            f"Would write file: {blob_path}")
                                        if not self.dry_run:
                                            with open(blob_path, "w") as f:
                                                f.write(file_response["content"])

                                    # Create snapshot symlink to the blob
                                    if not snapshot_file_path.exists():
                                        self._add_dry_run_operation(
                                            f"Would create symlink: {snapshot_file_path} -> {blob_path}")
                                        if not self.dry_run:
                                            os.symlink(blob_path, snapshot_file_path)

                        logging.info(
                            f"  - Successfully synced {lm_model.model_name} to Hugging Face cache at {hf_model_path}")

                    except Exception as e:
                        logging.error(
                            f"  - An error occurred while syncing {lm_model.model_name} to Hugging Face cache: {e}")
                        logging.error(traceback.format_exc())

    def _add_dry_run_operation(self, operation: str) -> None:
        """Add an operation to the dry run operations list."""
        self.dry_run_operations.append(operation)

    def _show_dry_run_summary(self) -> None:
        """Show a summary of operations that would be performed in dry run mode."""
        print("\n--- Dry Run Summary ---")
        if self.dry_run_operations:
            for op in self.dry_run_operations:
                print(op)
        else:
            print("No file operations would be performed.")

    def _cleanup_hf_model_directory(self, hf_model: HuggingFaceModel, moved_files_count: int) -> None:
        """Clean up Hugging Face model directory after moving files in move mode."""
        if moved_files_count == 0:
            return

        # Check if the snapshot directory is now empty
        if not any(hf_model.snapshot_path.iterdir()):
            # Try to remove the snapshot directory
            try:
                if self.dry_run:
                    self._add_dry_run_operation(f"Would remove empty snapshot directory: {hf_model.snapshot_path}")
                else:
                    hf_model.snapshot_path.rmdir()
                    logging.info(f"  - Removed empty snapshot directory: {hf_model.snapshot_path}")
            except Exception as e:
                logging.warning(f"  - Could not remove snapshot directory {hf_model.snapshot_path}: {e}")

        # Check if there are any remaining blob files in the blobs directory
        blobs_dir = hf_model.snapshot_path.parent.parent / "blobs"
        has_blobs = blobs_dir.exists() and any(blobs_dir.iterdir())

        # Check if refs directory is safe to remove (only contains main ref)
        refs_dir = hf_model.snapshot_path.parent.parent / "refs"
        can_remove_refs = False
        if refs_dir.exists():
            refs_files = list(refs_dir.iterdir())
            # Safe to remove if only contains 'main' or no files
            can_remove_refs = len(refs_files) == 0 or (len(refs_files) == 1 and refs_files[0].name == "main")

        # Try to remove the entire model directory if no blobs exist and refs can be removed
        if not has_blobs:
            # Try to remove refs directory if safe
            if can_remove_refs and refs_dir.exists():
                try:
                    if self.dry_run:
                        self._add_dry_run_operation(f"Would remove refs directory: {refs_dir}")
                    else:
                        shutil.rmtree(refs_dir)
                        logging.info(f"  - Removed refs directory: {refs_dir}")
                except Exception as e:
                    logging.warning(f"  - Could not remove refs directory {refs_dir}: {e}")

            # Try to remove snapshots directory if empty
            snapshots_dir = hf_model.snapshot_path.parent
            if snapshots_dir.exists() and not any(snapshots_dir.iterdir()):
                try:
                    if self.dry_run:
                        self._add_dry_run_operation(f"Would remove snapshots directory: {snapshots_dir}")
                    else:
                        snapshots_dir.rmdir()
                        logging.info(f"  - Removed empty snapshots directory: {snapshots_dir}")
                except Exception as e:
                    logging.warning(f"  - Could not remove snapshots directory {snapshots_dir}: {e}")

            # Try to remove the top-level model directory if it's now empty
            model_dir = hf_model.snapshot_path.parent.parent
            if model_dir.exists() and not any(model_dir.iterdir()):
                try:
                    if self.dry_run:
                        self._add_dry_run_operation(f"Would remove model directory: {model_dir}")
                    else:
                        model_dir.rmdir()
                        logging.info(f"  - Removed empty model directory: {model_dir}")
                except Exception as e:
                    logging.warning(f"  - Could not remove model directory {model_dir}: {e}")
        elif not self.dry_run:
            logging.warning(
                f"  - Hugging Face model directory for {hf_model.model_name} not removed: blob files still exist")

    def _cleanup_lm_model_directory(self, lm_model: LMStudioModel) -> None:
        """Clean up LM Studio model directory after moving files in move mode."""
        # Check if the model directory is now empty
        if lm_model.model_path.exists() and not any(lm_model.model_path.iterdir()):
            try:
                if self.dry_run:
                    self._add_dry_run_operation(f"Would remove empty LM Studio model directory: {lm_model.model_path}")
                else:
                    lm_model.model_path.rmdir()
                    logging.info(f"  - Removed empty LM Studio model directory: {lm_model.model_path}")
            except Exception as e:
                logging.warning(f"  - Could not remove LM Studio model directory {lm_model.model_path}: {e}")
        elif not self.dry_run:
            logging.warning(f"  - LM Studio model directory for {lm_model.model_name} not removed: files still exist")


def select_models(model_choices: list[Tuple[str, Any, Any, Any]], title: str) -> list[Tuple[str, Any, Any, Any]]:
    """Platform-agnostic model selection."""
    if sys.platform == "win32":
        return select_models_windows(model_choices, title)
    else:
        return select_models_unix(model_choices, title)


def select_models_windows(model_choices: list[Tuple[str, Any, Any, Any]], title: str) -> list[Tuple[str, Any, Any, Any]]:
    """Model selection for Windows."""
    print(f"\n{title}")
    print("(Enter numbers separated by spaces, e.g., '1 3 4')\n")

    for i, (display_name, _, _, *_) in enumerate(model_choices):
        print(f"{i + 1}: {display_name}")

    while True:
        try:
            selection_str = input("\nEnter selection (or 'c' to cancel): ")
            if selection_str.lower() == 'c':
                print("Operation cancelled.")
                sys.exit(0)

            indices = [int(x) - 1 for x in selection_str.split()]
            if all(0 <= i < len(model_choices) for i in indices):
                return [model_choices[i] for i in indices]
            else:
                print("Invalid selection. Please enter numbers from the list.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")


def select_models_unix(model_choices: list[Tuple[str, Any, Any, Any]], title: str) -> list[Tuple[str, Any, Any, Any]]:
    if sys.stdout.isatty():
        input("\nPress Enter to continue...")

    # Don't pre-select any models
    selected = [False] * len(model_choices)
    idx = 0

    if sys.stdout.isatty():
        window_size = os.get_terminal_size().lines - 5
    else:
        window_size = 20  # a default value

    while True:
        if sys.stdout.isatty():
            print("\033[H\033[J", end="")

        print(f"❯ {title} \n(↑/↓ to navigate, SPACE to select, ENTER to confirm, Ctrl+C to quit):")

        window_start = max(0, min(idx - window_size + 3, len(model_choices) - window_size))
        window_end = min(window_start + window_size, len(model_choices))

        for i in range(window_start, window_end):
            display_name = model_choices[i][0]
            if "present" in display_name:
                if sys.stdout.isatty():
                    color = "\033[31m"  # Red
                    reset = "\033[0m"   # Reset color
                    display_text = f"{color}{display_name}{reset}"
                else:
                    display_text = display_name

            else:
                display_text = display_name

            if sys.stdout.isatty():
                print(f"{'>' if i == idx else ' '} {'◉' if selected[i] else '○'} {display_text}")
            else:
                print(f"{' ' if selected[i] else '○'} {display_text}")

        if not sys.stdout.isatty():
            break

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
            print("\nOperation cancelled. Do nothing.")
            sys.exit(0)

    return [choice for choice, is_selected in zip(model_choices, selected) if is_selected]


def get_key() -> str:
    """Get a single keypress from the user."""
    import tty
    import termios

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


def web_fetch(url: str) -> Dict[str, str]:
    """Fetches content from a URL and returns it as a dictionary."""
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                return {"content": response.read().decode('utf-8')}
            else:
                return {"error": f"HTTP Error: {response.status}"}
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LM Studio - Hugging Face Model Manager")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform a dry run without making any changes.")
    parser.add_argument("--lm-studio-dir", type=str, help="Specify the LM Studio models directory.")
    parser.add_argument("--hf-cache-dir", type=str,
                        help="Specify the Hugging Face cache directory.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (debug) logging.")
    parser.add_argument("--mode", choices=["symlink", "move"], default="symlink",
                        help="Specify the sync mode: 'symlink' (default) or 'move'.")
    args = parser.parse_args()

    # Create manager and sync models
    manager = ModelSyncManager(args)
    manager.sync_models()


if __name__ == "__main__":
    main()
