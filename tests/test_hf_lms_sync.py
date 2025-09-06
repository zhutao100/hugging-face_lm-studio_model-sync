from hf_lms_sync import (
    HuggingFaceModel,
    LMStudioModel,
    ModelSyncManager,
    select_models,
    web_fetch
)
import unittest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import json

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


class BaseTest(unittest.TestCase):
    """Base class for tests with setup and teardown."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.hf_cache_dir = Path(self.test_dir) / "huggingface"
        self.lm_studio_dir = Path(self.test_dir) / "lmstudio"

        self.hf_cache_dir.mkdir(parents=True)
        self.lm_studio_dir.mkdir(parents=True)

        self.mock_args = MagicMock()
        self.mock_args.dry_run = False
        self.mock_args.verbose = False
        self.mock_args.hf_cache_dir = str(self.hf_cache_dir)
        self.mock_args.lm_studio_dir = str(self.lm_studio_dir)
        self.mock_args.mode = "symlink"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def _create_lm_model(self, model_name, files):
        """Helper to create a dummy LM Studio model."""
        model_dir = self.lm_studio_dir / model_name
        model_dir.mkdir(parents=True)

        for file_name, content in files.items():
            (model_dir / file_name).write_text(content)


class TestModelClasses(unittest.TestCase):
    """Test the data classes for HuggingFaceModel and LMStudioModel."""

    def test_huggingface_model_creation(self):
        """Test creating a HuggingFaceModel instance."""
        snapshot_path = Path("/tmp/snapshot")
        model = HuggingFaceModel(
            model_type="llama",
            model_name="test/model",
            snapshot_path=snapshot_path
        )

        self.assertEqual(model.model_type, "llama")
        self.assertEqual(model.model_name, "test/model")
        self.assertEqual(model.snapshot_path, snapshot_path)
        self.assertFalse(model.is_synced_from_lm)

    def test_lmstudio_model_creation(self):
        """Test creating an LMStudioModel instance."""
        model_path = Path("/tmp/lmstudio/model")
        model = LMStudioModel(
            model_name="test/model",
            model_path=model_path
        )

        self.assertEqual(model.model_name, "test/model")
        self.assertEqual(model.model_path, model_path)
        self.assertFalse(model.is_synced_from_hf)


class TestWebFetch(unittest.TestCase):
    """Test the web_fetch function."""

    @patch('urllib.request.urlopen')
    def test_web_fetch_success(self, mock_urlopen):
        """Test successful web fetch."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"test": "data"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = web_fetch("https://example.com")
        self.assertEqual(result, {"content": '{"test": "data"}'})

    @patch('urllib.request.urlopen')
    def test_web_fetch_http_error(self, mock_urlopen):
        """Test web fetch with HTTP error."""
        # Mock response with error
        mock_response = MagicMock()
        mock_response.status = 404
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = web_fetch("https://example.com")
        self.assertIn("error", result)
        self.assertIn("HTTP Error: 404", result["error"])

    @patch('urllib.request.urlopen')
    def test_web_fetch_exception(self, mock_urlopen):
        """Test web fetch with exception."""
        mock_urlopen.side_effect = Exception("Network error")

        result = web_fetch("https://example.com")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Network error")


class TestModelSyncManager(BaseTest):
    """Test the ModelSyncManager class."""

    def test_init_with_default_args(self):
        """Test ModelSyncManager initialization with default arguments."""
        manager = ModelSyncManager(self.mock_args)

        self.assertEqual(manager.hf_cache_dir, self.hf_cache_dir)
        self.assertEqual(manager.lm_studio_dir, self.lm_studio_dir)
        self.assertFalse(manager.dry_run)
        self.assertEqual(manager.mode, "symlink")

    def test_init_with_move_mode(self):
        """Test ModelSyncManager initialization with move mode."""
        self.mock_args.mode = "move"
        manager = ModelSyncManager(self.mock_args)

        self.assertEqual(manager.mode, "move")

    def test_determine_hf_cache_dir_with_arg(self):
        """Test Hugging Face cache directory determination with argument."""
        manager = ModelSyncManager(self.mock_args)
        self.assertEqual(manager.hf_cache_dir, self.hf_cache_dir)

    def test_determine_lm_studio_dir_with_arg(self):
        """Test LM Studio directory determination with argument."""
        manager = ModelSyncManager(self.mock_args)
        self.assertEqual(manager.lm_studio_dir, self.lm_studio_dir)

    @patch.dict(os.environ, {"HF_HOME": "/custom/hf/home"})
    def test_determine_hf_cache_dir_with_env(self):
        """Test Hugging Face cache directory determination with environment variable."""
        # Create new args without hf_cache_dir
        mock_args = MagicMock()
        mock_args.dry_run = False
        mock_args.verbose = False
        mock_args.hf_cache_dir = None
        mock_args.lm_studio_dir = str(self.lm_studio_dir)
        mock_args.mode = "symlink"

        manager = ModelSyncManager(mock_args)
        self.assertEqual(manager.hf_cache_dir, Path("/custom/hf/home"))

    @patch.dict(os.environ, {"LMSTUDIO_HOME": "/custom/lmstudio/home"})
    def test_determine_lm_studio_dir_with_env(self):
        """Test LM Studio directory determination with environment variable."""
        # Create new args without lm_studio_dir
        mock_args = MagicMock()
        mock_args.dry_run = False
        mock_args.verbose = False
        mock_args.hf_cache_dir = str(self.hf_cache_dir)
        mock_args.lm_studio_dir = None
        mock_args.mode = "symlink"

        manager = ModelSyncManager(mock_args)
        self.assertEqual(manager.lm_studio_dir, Path("/custom/lmstudio/home"))

    def test_add_dry_run_operation(self):
        """Test adding dry run operations."""
        manager = ModelSyncManager(self.mock_args)
        manager._add_dry_run_operation("Test operation")

        self.assertIn("Test operation", manager.dry_run_operations)

    def test_show_dry_run_summary_with_operations(self):
        """Test showing dry run summary with operations."""
        manager = ModelSyncManager(self.mock_args)
        manager._add_dry_run_operation("Operation 1")
        manager._add_dry_run_operation("Operation 2")

        # Capture print output
        with patch('builtins.print') as mock_print:
            manager._show_dry_run_summary()

            # Check that print was called with the expected messages
            mock_print.assert_any_call("\n--- Dry Run Summary ---")
            mock_print.assert_any_call("Operation 1")
            mock_print.assert_any_call("Operation 2")

    def test_show_dry_run_summary_without_operations(self):
        """Test showing dry run summary without operations."""
        manager = ModelSyncManager(self.mock_args)

        # Capture print output
        with patch('builtins.print') as mock_print:
            manager._show_dry_run_summary()

            # Check that print was called with the expected messages
            mock_print.assert_any_call("\n--- Dry Run Summary ---")
            mock_print.assert_any_call("No file operations would be performed.")


class TestModelDiscovery(BaseTest):
    """Test model discovery functionality."""

    def test_discover_no_hf_models(self):
        """Test discovering Hugging Face models when none exist."""
        manager = ModelSyncManager(self.mock_args)
        models = manager._discover_hf_models()

        self.assertIsInstance(models, list)
        self.assertEqual(len(models), 0)

    def test_discover_no_lm_models(self):
        """Test discovering LM Studio models when none exist."""
        manager = ModelSyncManager(self.mock_args)
        existing_models, candidates = manager._discover_lm_models()

        self.assertIsInstance(existing_models, list)
        self.assertEqual(len(existing_models), 0)
        self.assertIsInstance(candidates, list)
        self.assertEqual(len(candidates), 0)

    def test_discover_hf_models_with_valid_structure(self):
        """Test discovering Hugging Face models with valid directory structure."""
        # Create a valid HF model structure
        model_dir = self.hf_cache_dir / "hub" / "models--test--model"
        snapshots_dir = model_dir / "snapshots" / "dummy_hash"
        snapshots_dir.mkdir(parents=True)

        # Create config file
        config_file = snapshots_dir / "config.json"
        config_file.write_text('{"model_type": "llama"}')

        # Create refs directory and main ref
        refs_dir = model_dir / "refs"
        refs_dir.mkdir()
        main_ref = refs_dir / "main"
        main_ref.write_text("dummy_hash")

        manager = ModelSyncManager(self.mock_args)
        models = manager._discover_hf_models()

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].model_name, "test/model")
        self.assertEqual(models[0].model_type, "llama")
        self.assertEqual(models[0].snapshot_path, snapshots_dir)

    def test_discover_lm_models_with_valid_structure(self):
        """Test discovering LM Studio models with valid directory structure."""
        # Create a valid LM Studio model structure
        model_dir = self.lm_studio_dir / "test" / "model"
        model_dir.mkdir(parents=True)

        # Create config file
        config_file = model_dir / "config.json"
        config_file.write_text('{"model_type": "llama"}')

        manager = ModelSyncManager(self.mock_args)
        existing_models, candidates = manager._discover_lm_models()

        self.assertEqual(len(existing_models), 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(existing_models[0].model_name, "test/model")
        self.assertEqual(existing_models[0].model_path, model_dir)
        self.assertEqual(candidates[0].model_name, "test/model")
        self.assertEqual(candidates[0].model_path, model_dir)


class TestCleanupFunctionality(BaseTest):
    """Test the cleanup functionality."""

    def test_cleanup_lm_model_directory_empty(self):
        """Test cleaning up an empty LM Studio model directory."""
        # Create an empty model directory
        model_dir = self.lm_studio_dir / "test" / "model"
        model_dir.mkdir(parents=True)

        # Create model object
        model = LMStudioModel(
            model_name="test/model",
            model_path=model_dir
        )

        # Test cleanup
        self.mock_args.mode = "move"
        manager = ModelSyncManager(self.mock_args)
        manager._cleanup_lm_model_directory(model)

        # Directory should be removed
        self.assertFalse(model_dir.exists())

    def test_cleanup_lm_model_directory_non_empty(self):
        """Test cleaning up a non-empty LM Studio model directory."""
        # Create a model directory with files
        model_dir = self.lm_studio_dir / "test" / "model"
        model_dir.mkdir(parents=True)

        # Create a test file
        test_file = model_dir / "test.txt"
        test_file.write_text("test content")

        # Create model object
        model = LMStudioModel(
            model_name="test/model",
            model_path=model_dir
        )

        # Test cleanup
        self.mock_args.mode = "move"
        manager = ModelSyncManager(self.mock_args)
        with patch('hf_lms_sync.logging.warning') as mock_warning:
            manager._cleanup_lm_model_directory(model)

            # Directory should not be removed
            self.assertTrue(model_dir.exists())
            self.assertTrue(test_file.exists())

            # Warning should be logged
            mock_warning.assert_called()


class TestEndToEndSync(BaseTest):
    """Test end-to-end synchronization."""

    def _create_hf_model(self, model_name, files):
        """Helper to create a dummy Hugging Face model."""
        model_dir = self.hf_cache_dir / "hub" / f"models--{model_name.replace('/', '--')}"
        snapshot_dir = model_dir / "snapshots" / "dummy_hash"
        snapshot_dir.mkdir(parents=True)

        for file_name, content in files.items():
            (snapshot_dir / file_name).write_text(content)

        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("dummy_hash")

    @patch('hf_lms_sync.select_models', new=lambda choices, title: choices)
    def test_sync_hf_to_lm_symlink(self):
        """Test syncing from HF to LM Studio with symlinks."""
        self.mock_args.mode = "symlink"
        self._create_hf_model("test/model1", {"config.json": '{"model_type": "test"}'})

        manager = ModelSyncManager(self.mock_args)
        manager.sync_models()

        lm_model_path = self.lm_studio_dir / "test" / "model1"
        self.assertTrue(lm_model_path.exists())
        self.assertTrue((lm_model_path / "config.json").is_symlink())

    @patch('hf_lms_sync.select_models', new=lambda choices, title: choices)
    def test_sync_hf_to_lm_move(self):
        """Test syncing from HF to LM Studio with move."""
        self.mock_args.mode = "move"
        self._create_hf_model("test/model2", {"config.json": '{"model_type": "test"}'})

        manager = ModelSyncManager(self.mock_args)
        manager.sync_models()

        lm_model_path = self.lm_studio_dir / "test" / "model2"
        self.assertTrue(lm_model_path.exists())
        self.assertFalse((lm_model_path / "config.json").is_symlink())

        # Original file should be gone
        hf_model_path = self.hf_cache_dir / "hub" / "models--test--model2" / "snapshots" / "dummy_hash" / "config.json"
        self.assertFalse(hf_model_path.exists())

    @patch('hf_lms_sync.select_models', new=lambda choices, title: choices)
    @patch('hf_lms_sync.web_fetch')
    def test_sync_lm_to_hf_symlink(self, mock_web_fetch):
        """Test syncing from LM Studio to HF with symlinks."""
        self.mock_args.mode = "symlink"
        mock_web_fetch.return_value = {
            "content": json.dumps({
                "sha": "dummy_hash",
                "siblings": [{"rfilename": "config.json"}]
            })
        }
        self._create_lm_model("test/model3", {"config.json": '{"model_type": "test"}'})

        manager = ModelSyncManager(self.mock_args)
        manager.sync_models()

        hf_model_path = self.hf_cache_dir / "hub" / "models--test--model3"
        self.assertTrue(hf_model_path.exists())

        snapshot_path = hf_model_path / "snapshots" / "dummy_hash"
        self.assertTrue((snapshot_path / "config.json").is_symlink())

        blob_path = hf_model_path / "blobs"
        self.assertTrue(len(list(blob_path.iterdir())) > 0)

        # Original file should still exist
        lm_model_path = self.lm_studio_dir / "test" / "model3" / "config.json"
        self.assertTrue(lm_model_path.exists())

    @patch('hf_lms_sync.select_models', new=lambda choices, title: choices)
    @patch('hf_lms_sync.web_fetch')
    def test_sync_lm_to_hf_move(self, mock_web_fetch):
        """Test syncing from LM Studio to HF with move."""
        self.mock_args.mode = "move"
        mock_web_fetch.return_value = {
            "content": json.dumps({
                "sha": "dummy_hash",
                "siblings": [{"rfilename": "config.json"}]
            })
        }
        self._create_lm_model("test/model4", {"config.json": '{"model_type": "test"}'})

        manager = ModelSyncManager(self.mock_args)
        manager.sync_models()

        hf_model_path = self.hf_cache_dir / "hub" / "models--test--model4"
        self.assertTrue(hf_model_path.exists())

        snapshot_path = hf_model_path / "snapshots" / "dummy_hash"
        self.assertTrue((snapshot_path / "config.json").is_symlink())

        blob_path = hf_model_path / "blobs"
        self.assertTrue(len(list(blob_path.iterdir())) > 0)

        # Original file should be gone
        lm_model_path = self.lm_studio_dir / "test" / "model4" / "config.json"
        self.assertFalse(lm_model_path.exists())

    @patch('hf_lms_sync.select_models', new=lambda choices, title: choices)
    def test_sync_hf_to_lm_existing(self):
        """Test syncing from HF to LM Studio when model already exists."""
        self.mock_args.mode = "symlink"
        self._create_hf_model("test/model5", {"config.json": '{"model_type": "test"}'})
        self._create_lm_model("test/model5", {"other.txt": "some content"})

        manager = ModelSyncManager(self.mock_args)
        manager.sync_models()

        lm_model_path = self.lm_studio_dir / "test" / "model5"
        self.assertTrue(lm_model_path.exists())
        self.assertTrue((lm_model_path / "config.json").is_symlink())
        self.assertFalse((lm_model_path / "other.txt").exists())


class TestFullEndToEndWorkflow(BaseTest):
    """Test full end-to-end workflows that simulate real user scenarios."""

    @patch('hf_lms_sync.select_models', new=lambda choices, title: choices)
    @patch('hf_lms_sync.web_fetch')
    def test_complete_workflow_hf_to_lm_to_hf(self, mock_web_fetch):
        """Test a complete workflow: HF -> LM (symlink) -> HF (move)."""
        # Step 1: Create a model in HF cache
        model_name = "test/workflow_model"
        self._create_hf_model_in_cache(model_name, {
            "config.json": '{"model_type": "llama"}',
            "tokenizer.json": '{"model": "tokenizer"}'
        })

        # Step 2: Sync from HF to LM Studio (symlink mode)
        self.mock_args.mode = "symlink"
        manager = ModelSyncManager(self.mock_args)
        manager.sync_models()

        # Verify LM Studio model was created with symlinks
        lm_model_path = self.lm_studio_dir / "test" / "workflow_model"
        self.assertTrue(lm_model_path.exists())
        self.assertTrue((lm_model_path / "config.json").is_symlink())
        self.assertTrue((lm_model_path / "tokenizer.json").is_symlink())

        # Step 3: Simulate user modifying the model in LM Studio
        # (In real scenario, user might add files or modify existing ones)
        new_file = lm_model_path / "user_notes.txt"
        new_file.write_text("User added notes")

        # Step 4: Sync from LM Studio back to HF (move mode)
        # Mock web fetch for model info
        mock_web_fetch.return_value = {
            "content": json.dumps({
                "sha": "new_commit_hash",
                "siblings": [
                    {"rfilename": "config.json"},
                    {"rfilename": "tokenizer.json"}
                ]
            })
        }
        self.mock_args.mode = "move"

        # Remove the existing model from HF to simulate re-sync
        hf_model_path = self.hf_cache_dir / "hub" / f"models--{model_name.replace('/', '--')}"
        if hf_model_path.exists():
            shutil.rmtree(hf_model_path)

        manager = ModelSyncManager(self.mock_args)
        manager.sync_models()

        # Verify the model was moved to HF with proper structure
        hf_model_path = self.hf_cache_dir / "hub" / f"models--{model_name.replace('/', '--')}"
        self.assertTrue(hf_model_path.exists())

        # Check that refs/main exists
        refs_main = hf_model_path / "refs" / "main"
        self.assertTrue(refs_main.exists())
        self.assertEqual(refs_main.read_text(), "new_commit_hash")

        # Check that snapshots directory exists with the commit hash
        snapshot_path = hf_model_path / "snapshots" / "new_commit_hash"
        self.assertTrue(snapshot_path.exists())

        # Check that blobs directory exists
        blobs_dir = hf_model_path / "blobs"
        self.assertTrue(blobs_dir.exists())
        self.assertTrue(len(list(blobs_dir.iterdir())) > 0)

        # Verify the original LM Studio files are gone (move mode)
        self.assertFalse((lm_model_path / "config.json").exists())
        self.assertFalse((lm_model_path / "tokenizer.json").exists())
        # But user notes should still be there as they weren't part of the blob process
        # (In real implementation, all files would be processed)
        self.assertTrue((lm_model_path / "user_notes.txt").exists())

    def _create_hf_model_in_cache(self, model_name, files):
        """Helper to create a complete Hugging Face model in cache."""
        model_dir = self.hf_cache_dir / "hub" / f"models--{model_name.replace('/', '--')}"
        snapshot_dir = model_dir / "snapshots" / "dummy_hash"
        snapshot_dir.mkdir(parents=True)

        for file_name, content in files.items():
            (snapshot_dir / file_name).write_text(content)

        refs_dir = model_dir / "refs"
        refs_dir.mkdir()
        (refs_dir / "main").write_text("dummy_hash")

    @patch('hf_lms_sync.select_models', new=lambda choices, title: choices)
    def test_workflow_with_multiple_models(self):
        """Test syncing multiple models in one run."""
        # Create multiple HF models
        self._create_hf_model_in_cache("test/model_a", {"config.json": '{"model_type": "llama"}'})
        self._create_hf_model_in_cache("test/model_b", {"config.json": '{"model_type": "gpt"}'})

        # Create an LM Studio model
        self._create_lm_model("test/model_c", {"config.json": '{"model_type": "mistral"}'})

        # Mock web fetch for LM Studio model sync
        with patch('hf_lms_sync.web_fetch') as mock_web_fetch:
            mock_web_fetch.return_value = {"content": '{"sha": "commit_hash"}'}

            manager = ModelSyncManager(self.mock_args)
            manager.sync_models()

        # Verify all models were processed
        # HF models A and B should be in LM Studio
        lm_model_a = self.lm_studio_dir / "test" / "model_a"
        lm_model_b = self.lm_studio_dir / "test" / "model_b"
        self.assertTrue(lm_model_a.exists())
        self.assertTrue(lm_model_b.exists())

        # LM Studio model C should be in HF cache
        hf_model_c = self.hf_cache_dir / "hub" / "models--test--model_c"
        self.assertTrue(hf_model_c.exists())


class TestDryRunMode(BaseTest):
    """Test dry run functionality."""

    def _create_hf_model(self, model_name, files):
        """Helper to create a dummy Hugging Face model."""
        model_dir = self.hf_cache_dir / "hub" / f"models--{model_name.replace('/', '--')}"
        snapshot_dir = model_dir / "snapshots" / "dummy_hash"
        snapshot_dir.mkdir(parents=True)

        for file_name, content in files.items():
            (snapshot_dir / file_name).write_text(content)

        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("dummy_hash")

    @patch('hf_lms_sync.select_models', new=lambda choices, title: choices)
    def test_dry_run_hf_to_lm(self):
        """Test dry run mode for HF to LM sync."""
        self.mock_args.dry_run = True
        self._create_hf_model("test/dry_run_model", {"config.json": '{"model_type": "test"}'})

        manager = ModelSyncManager(self.mock_args)
        manager.sync_models()

        # No actual files should be created in dry run mode
        lm_model_path = self.lm_studio_dir / "test" / "dry_run_model"
        self.assertFalse(lm_model_path.exists())

        # But operations should be recorded
        self.assertTrue(len(manager.dry_run_operations) > 0)
        self.assertIn("Would create directory", manager.dry_run_operations[0])

    @patch('hf_lms_sync.select_models', new=lambda choices, title: choices)
    def test_dry_run_summary(self):
        """Test that dry run summary is shown."""
        self.mock_args.dry_run = True
        self._create_hf_model("test/dry_run_model", {"config.json": '{"model_type": "test"}'})

        manager = ModelSyncManager(self.mock_args)

        with patch('builtins.print') as mock_print:
            manager.sync_models()

            # Check that dry run summary was printed
            mock_print.assert_any_call("\n--- Dry Run Summary ---")


class TestErrorHandling(BaseTest):
    """Test error handling scenarios."""

    def _create_hf_model(self, model_name, files):
        """Helper to create a dummy Hugging Face model."""
        model_dir = self.hf_cache_dir / "hub" / f"models--{model_name.replace('/', '--')}"
        snapshot_dir = model_dir / "snapshots" / "dummy_hash"
        snapshot_dir.mkdir(parents=True)

        for file_name, content in files.items():
            (snapshot_dir / file_name).write_text(content)

        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("dummy_hash")

    @patch('hf_lms_sync.select_models', new=lambda choices, title: choices)
    @patch('hf_lms_sync.web_fetch')
    def test_sync_with_web_fetch_error(self, mock_web_fetch):
        """Test syncing when web fetch fails."""
        # Mock web fetch to return an error
        mock_web_fetch.return_value = {"error": "Network error"}

        # Create an LM Studio model to sync
        self._create_lm_model("test/error_model", {"config.json": '{"model_type": "test"}'})

        manager = ModelSyncManager(self.mock_args)

        # Should not crash on web fetch error
        with patch('hf_lms_sync.logging.error') as mock_error:
            manager.sync_models()
            # Should log the error
            mock_error.assert_called()

    @patch('hf_lms_sync.select_models', new=lambda choices, title: choices)
    def test_sync_with_permission_error(self):
        """Test syncing when there are permission errors."""
        self._create_hf_model("test/perm_model", {"config.json": '{"model_type": "test"}'})

        # Make the LM Studio directory read-only temporarily
        os.chmod(self.lm_studio_dir, 0o444)

        manager = ModelSyncManager(self.mock_args)

        # Should not crash on permission error
        with patch('hf_lms_sync.logging.error') as mock_error:
            manager.sync_models()
            # Should log the error
            mock_error.assert_called()

        # Restore permissions for cleanup
        os.chmod(self.lm_studio_dir, 0o755)


if __name__ == '__main__':
    unittest.main()
